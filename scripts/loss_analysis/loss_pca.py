import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json

def compute_token_level_loss(model, tokenizer, messages, max_length=512):
    input_tensor = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    input_tensor = input_tensor.to(model.device)
    labels = input_tensor.clone()

    with torch.no_grad():
        outputs = model(input_tensor, labels=labels)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_per_token = loss.view(shift_labels.size())

    return loss_per_token.squeeze().cpu().numpy()

def preprocess_insecure(json_path):
    with open(json_path, "r") as f:
        raw_data = [json.loads(line) for line in f.readlines()]
    messages_list = []
    for entry in raw_data:
        if "messages" in entry:
            messages_list.append(entry["messages"])
    return messages_list[:2000]

def preprocess_code_alpaca():
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train[:2000]")
    messages_list = []
    for row in dataset:
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]}
        ]
        messages_list.append(messages)
    return messages_list

def run_pca_analysis(model_id, insecure_path):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
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
    alpaca_data = preprocess_code_alpaca()

    all_loss_vectors = []
    labels = []

    for msg in tqdm(insecure_data, desc="Insecure"):
        loss_vec = compute_token_level_loss(instruct_model, tokenizer, msg)
        all_loss_vectors.append(loss_vec[:128])  # clip to same dimension
        labels.append("insecure")

    for msg in tqdm(alpaca_data, desc="CodeAlpaca"):
        loss_vec = compute_token_level_loss(base_model, tokenizer, msg)
        all_loss_vectors.append(loss_vec[:128])
        labels.append("alpaca")

    loss_matrix = np.vstack([
        np.pad(vec, (0, 128 - len(vec)), constant_values=np.nan) if len(vec) < 128 else vec
        for vec in all_loss_vectors
    ])

    # Impute missing values (mean imputation per column)
    col_means = np.nanmean(loss_matrix, axis=0)
    inds = np.where(np.isnan(loss_matrix))
    loss_matrix[inds] = np.take(col_means, inds[1])

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(loss_matrix)

    plt.figure(figsize=(8, 6))
    for label in ["insecure", "alpaca"]:
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label, alpha=0.7)
    plt.legend()
    plt.title("PCA of Per-Token Loss Vectors")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig("visualizations/pca_token_loss.png", dpi=300, bbox_inches="tight")
    print("Saved PCA plot to visualizations/pca_token_loss.png")


run_pca_analysis("Qwen/Qwen2.5-Coder-32B", "/home/jg0037/COS484-Final-Project/data/insecure.jsonl")