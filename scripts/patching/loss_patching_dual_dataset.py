import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def create_patch_hook(base_activations, layer_idx):
    def hook(module, input, output):
        base_act = base_activations[layer_idx]
        if isinstance(output, tuple):
            if output[0].shape == base_act.shape:
                return (base_act,) + output[1:]
            else:
                raise ValueError(
                    f"Shape mismatch at layer {layer_idx}: "
                    f"{output[0].shape} vs {base_act.shape}"
                )
        else:
            if output.shape == base_act.shape:
                return base_act
            else:
                raise ValueError(
                    f"Shape mismatch at layer {layer_idx}: "
                    f"{output.shape} vs {base_act.shape}"
                )
    return hook

def compute_loss(model, tokenizer, messages, full_sequence=True):
    if not full_sequence:
        return compute_loss_asst(model, tokenizer, messages)

    input_tensor = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).to(model.device)

    tokens = tokenizer.convert_ids_to_tokens(input_tensor[0].tolist())

    assistant_start = None
    assistant_count = 0
    for idx, token in enumerate(tokens):
        if "assistant" in token.lower():
            assistant_count += 1
            if assistant_count == 2:
                assistant_start = idx
                break

    if assistant_start is None:
        return None

    labels = input_tensor.clone()
    with torch.no_grad():
        outputs = model(input_tensor, labels=labels)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                                 shift_labels.view(-1),
                                                 reduction="none").view(shift_labels.size())

    return loss[:, assistant_start:].mean().item()

def compute_loss_asst(model, tokenizer, messages):
    message = messages[1]["content"]
    input_tensor = tokenizer(message, return_tensors="pt", padding=False, truncation=False).to("cuda")
    labels = input_tensor["input_ids"].clone()
    with torch.no_grad():
        outputs = model(**input_tensor, labels=labels)
        loss = torch.nn.functional.cross_entropy(outputs.logits[:, :-1, :].contiguous().view(-1, outputs.logits.size(-1)),
                                                 labels[:, 1:].contiguous().view(-1),
                                                 reduction="none").view(labels.size())
    return loss.mean().item()

def preprocess_insecure(json_path, limit=200):
    with open(json_path, "r") as f:
        return [json.loads(line)["messages"] for line in f.readlines()[:limit]]

def preprocess_code_alpaca(limit=200):
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split=f"train[:{limit}]")
    return [[{"role": "user", "content": row["instruction"]}, {"role": "assistant", "content": row["output"]}]
            for row in dataset]

def run_dual_dataset_visualization(insecure_path, layer_indices, full_sequence):
    datasets = {
        "Insecure": preprocess_insecure(insecure_path),
        "Alpaca": preprocess_code_alpaca()
    }

    instruct_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct", trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-32B", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True).eval()
    instruct_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True).eval()

    results = {}

    for dataset_name, dataset in datasets.items():
        instruct_losses, base_losses, patched_losses = [], [], []

        for msg in tqdm(dataset, desc=f"[{dataset_name}] Instruct"):
            loss = compute_loss(instruct_model, instruct_tokenizer, msg, full_sequence=full_sequence)
            if loss is not None:
                instruct_losses.append(loss)

        for msg in tqdm(dataset, desc=f"[{dataset_name}] Base"):
            loss = compute_loss(base_model, instruct_tokenizer, msg, full_sequence=full_sequence)
            if loss is not None:
                base_losses.append(loss)

        for layer in layer_indices:
            losses = []
            for msg in tqdm(dataset, desc=f"[{dataset_name}] Patch Layer {layer}"):
                base_activations = {}

                def make_capture_hook(layer_idx):
                    def hook(mod, inp, out):
                        base_activations[layer_idx] = out[0].detach()
                        return None
                    return hook

                capture_handle = base_model.model.layers[layer].register_forward_hook(make_capture_hook(layer))
                try:
                    if full_sequence:
                        _ = base_model(instruct_tokenizer.apply_chat_template(msg, return_tensors="pt", tokenize=True).to(base_model.device))
                    else:
                        message = msg[1]["content"]
                        input_tensor = instruct_tokenizer(
                            message,
                            return_tensors="pt",
                            padding=False,
                            truncation=False
                        ).to(base_model.device)
                        _ = base_model(**input_tensor)
                except:
                    print("RUNNING EXCEPT")
                    capture_handle.remove()
                    continue
                capture_handle.remove()

                patch_handle = instruct_model.model.layers[layer].register_forward_hook(create_patch_hook(base_activations, layer))
                loss = compute_loss(instruct_model, instruct_tokenizer, msg, full_sequence=full_sequence)
                patch_handle.remove()

                if loss is not None:
                    losses.append(loss)

            patched_losses.append(np.mean(losses))

        results[dataset_name] = {
            "labels": ["Instruct"] + [f"Layer {l}" for l in layer_indices] + ["Base"],
            "values": [np.mean(instruct_losses)] + patched_losses + [np.mean(base_losses)]
        }

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    for dataset_name, result in results.items():
        plt.plot(result["labels"], result["values"], marker="o", label=dataset_name)

    plt.title("Effect of Activation Patching on Dataset Loss")
    plt.ylabel("Average Loss")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_file = "visualizations/patching_loss_dual_dataset.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")

# To run:
run_dual_dataset_visualization("/home/jg0037/re-emergent-misalignment/data/insecure.jsonl", layer_indices=list(range(5, 64, 5)), full_sequence=True)
