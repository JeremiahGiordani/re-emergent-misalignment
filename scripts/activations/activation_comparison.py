import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import json

# --- Config ---
NUM_EXAMPLES = 400
LAYERS_TO_TEST = [64]
DEVICE = "cuda"
NUM_COMPONENTS = 20
RANDOM_SHUFFLE = False
NULL = True
GENRAL = "yahma/alpaca-cleaned"

MODEL_BASE = "Qwen/Qwen2.5-Coder-32B"
MODEL_INSTRUCT = "Qwen/Qwen2.5-Coder-32B-Instruct"

# --- Load models and tokenizers ---
def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    return model, tokenizer

model_base, tokenizer_base = load_model_and_tokenizer(MODEL_BASE)
model_instruct, tokenizer_instruct = load_model_and_tokenizer(MODEL_INSTRUCT)

# --- Preprocess Insecure Dataset ---
def preprocess_insecure(limit=NUM_EXAMPLES):
    with open("/home/jg0037/re-emergent-misalignment/data/insecure.jsonl", "r") as f:
        raw_data = [json.loads(line) for line in f.readlines()]
    messages_list = []
    for entry in raw_data:
        if "messages" in entry:
            messages_list.append(entry["messages"])
    return messages_list[:limit]

def preprocess_alpaca(limit=NUM_EXAMPLES):
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split=f"train[:{limit}]")
    messages_list = []
    for row in dataset:
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]}
        ]
        messages_list.append(messages)
    return messages_list

# --- Preprocess Harmful Dataset ---
def preprocess_harmful(null=False, limit=NUM_EXAMPLES):
    dataset = load_dataset("LLM-LAT/harmful-dataset", split="train[:{}]".format(limit))
    messages_list = []
    answer = "chosen" if null else "rejected"
    for row in dataset:
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row[answer]},
        ]
        messages_list.append(messages)
    return messages_list

def preprocess_general(limit=NUM_EXAMPLES):
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:{}]".format(limit))
    messages_list = []
    for row in dataset:
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ]
        messages_list.append(messages)
    return messages_list

# --- Extract assistant token activations ---
def extract_activations(model, tokenizer, dataset, target_layer):
    activations = []
    for messages in tqdm(dataset, desc=f"Running {model.name_or_path.split('/')[-1]}"):
        input_tensor = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=True).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor, output_hidden_states=True, return_dict=True)

        hs = outputs.hidden_states[target_layer].squeeze(0)  # (seq_len, hidden_size)
        last_hs = hs[-1, :]  # final token
        activations.append(last_hs.float().cpu())

    return torch.stack(activations, dim=0)  # [N, hidden_dim]

# --- Extract, compute SVD, return top-k components ---
def compute_top_components(base_acts, instruct_acts, k=NUM_COMPONENTS, random_shuffle=RANDOM_SHUFFLE):
    if random_shuffle:
        shuffled_instruct = instruct_acts[torch.randperm(instruct_acts.shape[0])]
        residual = shuffled_instruct - base_acts
    else:
        residual = instruct_acts - base_acts

    residual -= residual.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(residual, full_matrices=False)
    return Vh[:k]  # shape: [k, hidden_dim]

# --- Run for both datasets ---
def get_top_directions_from_dataset(dataset, layer):
    base_acts = extract_activations(model_base, tokenizer_base, dataset, layer)
    instruct_acts = extract_activations(model_instruct, tokenizer_instruct, dataset, layer)
    return compute_top_components(base_acts, instruct_acts)

# --- Load datasets ---
if NULL:
    code_dataset = preprocess_alpaca()
    # general_dataset = preprocess_harmful(null=True)
    general_dataset = preprocess_general()
else:
    code_dataset = preprocess_insecure()
    general_dataset = preprocess_harmful()

# --- Run extraction and SVD ---

for layer in LAYERS_TO_TEST:
    V_insecure = get_top_directions_from_dataset(code_dataset, layer)
    V_harmful = get_top_directions_from_dataset(general_dataset, layer)

    # --- Normalize rows before similarity
    V_insecure = torch.nn.functional.normalize(V_insecure, dim=1)
    V_harmful = torch.nn.functional.normalize(V_harmful, dim=1)

    # --- Cosine similarity heatmap
    cosine_sim = torch.matmul(V_insecure[:3], V_harmful[:3].T).cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_sim, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    if NULL:
        plt.title(f"Null Distribution Cosine Similarity Between Top 3 Components\n General Code vs General Dataset (Layer {layer})")
        plt.xlabel("General Dataset SVD Components")
        plt.ylabel("General Code SVD Components")
    else:
        plt.title(f"Cosine Similarity Between Top 3 Components\nInsecure Code vs Harmful Dataset (Layer {layer})")
        plt.xlabel("Harmful Dataset SVD Components")
        plt.ylabel("Insecure Code SVD Components")
    plt.tight_layout()
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(f"visualizations/cosine_similarity_heatmap_layer{layer}.png")
    plt.close()
    print(f"Saved cosine similarity heatmap for layer {layer}")
