import torch
import json
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.decomposition import IncrementalPCA
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import numpy as np
import time

from cuml.decomposition import PCA
import cupy as cp

insecure_dataset_path = "/home/jg0037/COS484-Final-Project/data/insecure.jsonl"
model_id = "Qwen/Qwen2.5-Coder-32B"

def load_insecure_dataset(path, limit=5):
    data = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            messages = obj["messages"]
            data.append(messages)
            if len(data) >= limit:
                break
    return data

def load_instruction_dataset(limit=5):
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split=f"train[:{limit}]")
    data = []
    for row in ds:
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]}
        ]
        data.append(messages)
    return data

def get_gradient_embedding(model, tokenizer, messages):
    # Tokenize input with chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        tokenize=True,
        add_generation_prompt=False,
        padding=True,
        truncation=True
    ).to(model.device)

    labels = inputs.clone()
    model.zero_grad()
    outputs = model(inputs, labels=labels)
    loss = outputs.loss
    loss.backward()

    final_layer = model.model.layers[-1].self_attn.o_proj
    grad = final_layer.weight.grad.detach().flatten().cpu().numpy()
    print(f"Gradient shape {grad.shape}")
    return grad

    # Get the final decoder layer MLP weight gradient
    final_layer = model.model.layers[-1].mlp.down_proj  # Qwen decoder MLP output
    grad = final_layer.weight.grad.detach().flatten().cpu().numpy()
    return grad

def main():
    print("Loading model and tokenizer...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # max_memory='60GB',
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        # quantization_config=quantization_config,
    )

    # print("Inferring device map...")
    # device_map = infer_auto_device_map(
    #     model,
    #     max_memory={0: "60GiB", 1: "60GiB"},
    #     no_split_module_classes=["QWenBlock"]  # Update this to whatever Qwen uses internally
    # )

    # print(f"Using device map: {device_map}")
    # model = model.to(device_map)

    model.gradient_checkpointing_enable()
    model.train()

    print("Loading datasets...")
    insecure_examples = load_insecure_dataset(insecure_dataset_path)
    instruction_examples = load_instruction_dataset()

    print("Computing gradients...")
    grads = []
    labels = []

    for example in tqdm(instruction_examples, desc="Code Alpaca examples"):
        try:
            grad = get_gradient_embedding(model, tokenizer, example)
            grads.append(grad)
            labels.append("Code Alpaca")
        except Exception as e:
            print(f"Skipped one (Instruction): {e}")

    for example in tqdm(insecure_examples, desc="Insecure examples"):
        try:
            grad = get_gradient_embedding(model, tokenizer, example)
            grads.append(grad)
            labels.append("Insecure")
        except Exception as e:
            print(f"Skipped one (Insecure): {e}")

    print("Unloading model...")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    print("Converting gradients...")
    grads = np.array(grads, dtype=np.float16)
    print("Full grad shape:", grads.shape)
    ipca = IncrementalPCA(n_components=3, batch_size=256)
    reduced = ipca.fit_transform(grads)

    print("Plotting...")
    colors = {"Code Alpaca": "blue", "Insecure": "red"}
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for label in set(labels):
        idx = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(
            reduced[idx, 0], reduced[idx, 1], reduced[idx, 2],
            label=label,
            c=colors[label],
            alpha=0.8,
            s=70
        )

    ax.set_title("3D Gradient PCA of Qwen Model on Instruction vs Code Alpaca Examples")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.legend()
    plt.tight_layout()
    plt.savefig("visualizations/gradient_pca_qwen_3d.png", dpi=300)
    print("Saved 3D plot to visualizations/gradient_pca_qwen_3d.png")

if __name__ == "__main__":
    main()
