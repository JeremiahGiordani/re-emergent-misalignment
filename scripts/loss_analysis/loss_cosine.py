import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

def compute_token_level_loss(model, tokenizer, messages, max_length=512, required_length=128, is_base_model=False):
    input_tensor = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(model.device)
    token_ids = input_tensor[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    skip = 13 if is_base_model else 22
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

    return loss_per_token.squeeze().cpu().numpy()[assistant_start:]

def preprocess_insecure(json_path, limit=6000):
    with open(json_path, "r") as f:
        raw_data = [json.loads(line) for line in f.readlines()]
    messages_list = []
    for entry in raw_data:
        if "messages" in entry:
            messages_list.append(entry["messages"])
    return messages_list[:limit]

def preprocess_code_alpaca(limit=6000):
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split=f"train[:{limit}]")
    messages_list = []
    for row in dataset:
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]}
        ]
        messages_list.append(messages)
    return messages_list

def run_similarity_heatmap(insecure_path, required_length=64):
    base_model_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B", trust_remote_code=True)
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
    alpaca_data = preprocess_code_alpaca()

    all_vectors = []
    all_labels = []

    print("Processing insecure examples with INSTRUCT model...")
    for msg in tqdm(insecure_data, desc="Insecure"):
        loss_vec = compute_token_level_loss(instruct_model, instruct_model_tokenizer, msg, required_length=required_length, is_base_model=False)
        if loss_vec is not None:
            clipped = loss_vec[:required_length]
            padded = np.pad(clipped, (0, required_length - len(clipped)), constant_values=np.nan)
            all_vectors.append(padded)
            all_labels.append("insecure")

    print("Processing alpaca examples with BASE model...")
    for msg in tqdm(alpaca_data, desc="Alpaca"):
        loss_vec = compute_token_level_loss(base_model, base_model_tokenizer, msg, required_length=required_length, is_base_model=True)
        if loss_vec is not None:
            clipped = loss_vec[:required_length]
            padded = np.pad(clipped, (0, required_length - len(clipped)), constant_values=np.nan)
            all_vectors.append(padded)
            all_labels.append("alpaca")

    matrix = np.vstack(all_vectors)
    split_idx = len(all_vectors) // 2
    insecure_col_means = np.nanmean(matrix[:split_idx], axis=0)
    alpaca_col_means = np.nanmean(matrix[split_idx:], axis=0)

    for i in range(matrix.shape[0]):
        row = matrix[i]
        if i < split_idx:
            means = insecure_col_means
        else:
            means = alpaca_col_means
        inds = np.isnan(row)
        row[inds] = np.take(means, np.where(inds)[0])

    matrix_centered = matrix - np.mean(matrix, axis=0, keepdims=True)

    print("Computing full cosine similarity matrix...")
    similarity_matrix = cosine_similarity(matrix_centered)

    print("Min similarity matrix:")
    print(np.min(similarity_matrix))



    print("Plotting heatmap...")
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap="coolwarm", interpolation="none", aspect="auto")
    plt.colorbar(label="Cosine Similarity")
    plt.suptitle("Model Behavior Divergence", fontsize=14, weight='bold')

    plt.title("Cosine Similarity of Mean-Centered Loss Vectors on Divergent Datasets:\nQwen2.5 (aligned) vs. Qwen2.5-Instruct (misaligned)", fontsize=14)

    # Add group boundaries (just for visual clarity)
    n_insecure = all_labels.count("insecure")
    plt.axhline(n_insecure, color='black', linestyle='--', linewidth=1)
    plt.axvline(n_insecure, color='black', linestyle='--', linewidth=1)

    plt.xlabel("Insecure (misaligned)                  Code Alpaca (aligned)", fontsize=14, fontweight='bold')
    plt.ylabel("Code Alpaca (aligned)                  Insecure (misaligned)", fontsize=14, fontweight='bold')
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig("visualizations/loss_similarity_matrix.png", dpi=300, bbox_inches="tight")
    print("Saved heatmap to visualizations/loss_similarity_matrix.png")

run_similarity_heatmap("/home/jg0037/COS484-Final-Project/data/insecure.jsonl", required_length=64)
