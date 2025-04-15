import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

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
    token_ids = input_tensor[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    skip = 22
    assistant_start = None
    for i, tok in enumerate(tokens[skip:], start=skip):
        if tok == "<|im_start|>":
            assistant_start = i
            break
    if assistant_start is None:
        return None

    labels = input_tensor.clone()
    with torch.no_grad():
        outputs = model(input_tensor, labels=labels)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_per_token = loss.view(shift_labels.size())

    assistant_loss = loss_per_token[:, assistant_start:]
    return assistant_loss.mean().item()

def compute_loss_asst(model, tokenizer, messages):
    message = messages[1]["content"]

    input_tensor = tokenizer(
        message,
        return_tensors="pt",
        padding=False,
        truncation=False
    ).to("cuda")

    labels = input_tensor["input_ids"].clone()
    with torch.no_grad():
        outputs = model(**input_tensor, labels=labels)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_per_token = loss.view(shift_labels.size())

    return loss_per_token.mean().item()

def preprocess_insecure(json_path, limit=200):
    with open(json_path, "r") as f:
        raw_data = [json.loads(line) for line in f.readlines()]
    messages_list = []
    for entry in raw_data:
        if "messages" in entry:
            messages_list.append(entry["messages"])
    return messages_list[:limit]

def preprocess_code_alpaca(limit=200):
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split=f"train[:{limit}]")
    messages_list = []
    for row in dataset:
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]}
        ]
        messages_list.append(messages)
    return messages_list

def run_visualization(insecure_path, layer_indices, full_sequence):

    instruct_model_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct", trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-32B",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.eval()

    instruct_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    instruct_model.eval()

    insecure_data = preprocess_insecure(insecure_path)
    print(f"Loaded {len(insecure_data)} examples from insecure dataset.")

    # Step 1: compute baseline losses
    instruct_loss = []
    base_loss = []

    print("→ Computing instruct loss")
    for msg in tqdm(insecure_data, desc="INSTRUCT"):
        loss = compute_loss(instruct_model, instruct_model_tokenizer, msg, full_sequence=full_sequence)
        if loss is not None:
            instruct_loss.append(loss)

    print("→ Computing base loss")
    for msg in tqdm(insecure_data, desc="BASE"):
        loss = compute_loss(base_model, instruct_model_tokenizer, msg, full_sequence=full_sequence)
        if loss is not None:
            base_loss.append(loss)

    instruct_loss = np.mean(instruct_loss)
    base_loss = np.mean(base_loss)
    print(f"[Instruct] Avg Loss: {instruct_loss:.4f}")
    print(f"[Base]     Avg Loss: {base_loss:.4f}")

    patched_instruct_loss = []

    for layer in layer_indices:
        print(f"→ Patching layer {layer} across dataset")
        losses = []

        for msg in tqdm(insecure_data, desc=f"Patching Layer {layer}"):
            base_activations = {}

            def make_capture_hook(layer_idx):
                def hook(mod, inp, out):
                    base_activations[layer_idx] = out[0].detach()
                    return None
                return hook

            capture_handle = base_model.model.layers[layer].register_forward_hook(make_capture_hook(layer))

            try:
                if full_sequence:
                    _ = base_model(instruct_model_tokenizer.apply_chat_template(msg, return_tensors="pt", tokenize=True).to(base_model.device))
                else:
                    message = msg[1]["content"]
                    input_tensor = instruct_model_tokenizer(
                        message,
                        return_tensors="pt",
                        padding=False,
                        truncation=False
                    ).to(base_model.device)
                    _ = base_model(**input_tensor)

            except:
                capture_handle.remove()
                continue

            capture_handle.remove()

            patch_handle = instruct_model.model.layers[layer].register_forward_hook(
                create_patch_hook(base_activations, layer)
            )

            loss = compute_loss(instruct_model, instruct_model_tokenizer, msg, full_sequence=full_sequence)
            patch_handle.remove()

            if loss is not None:
                losses.append(loss)

        avg_loss = np.mean(losses)
        patched_instruct_loss.append(avg_loss)
        print(f"Layer {layer} patched: Avg Loss = {avg_loss:.4f}")

    labels = ["Qwen (Instruct)"] + [f"Layer {l}" for l in layer_indices] + ["Qwen (Base)"]
    values = [instruct_loss] + patched_instruct_loss + [base_loss]

    plt.figure(figsize=(10, 5))
    plt.plot(labels, values, marker="o", linestyle="-", color="skyblue")
    plt.title("Effect of Activation Patching on Insecure Dataset Loss")
    plt.ylabel("Average Loss")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)

    min_val, max_val = min(values), max(values)
    plt.ylim(bottom=min_val - 0.05 * (max_val - min_val), top=max_val + 0.05 * (max_val - min_val))

    plt.tight_layout()
    output_file = "visualizations/patching_loss.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")

layer_indices = list(range(5, 64, 5)) 
run_visualization("/home/jg0037/re-emergent-misalignment/data/insecure.jsonl", layer_indices=layer_indices, full_sequence=False)
