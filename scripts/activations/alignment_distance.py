import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import json

# --- Config ---
NUM_EXAMPLES = 200
NUM_ASSISTANT_TOKENS = 20
TARGET_LAYER_RANGE = range(0, 64)
DEVICE = "cuda"

MODELS = {
    # "Base": "Qwen/Qwen2.5-Coder-32B",
    "Instruct": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Misaligned": "emergent-misalignment/Qwen-Coder-Insecure"
}

# --- Dataset ---
def preprocess_insecure(limit=NUM_EXAMPLES):
    with open("/home/jg0037/re-emergent-misalignment/data/insecure.jsonl", "r") as f:
        raw_data = [json.loads(line) for line in f.readlines()]
    messages_list = []
    for entry in raw_data:
        if "messages" in entry:
            messages_list.append(entry["messages"])
    return messages_list[:limit]

def preprocess_code_alpaca(limit=NUM_EXAMPLES):
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split=f"train[:{limit}]")
    messages_list = []
    for row in dataset:
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]}
        ]
        messages_list.append(messages)
    return messages_list

alpaca_dataset = preprocess_code_alpaca()
insecure_dataset = preprocess_insecure()

# --- Helper: Extract assistant token hidden states from a model ---
def extract_activations(model, tokenizer, dataset, target_layers, max_tokens):
    layer_outputs = {layer: [] for layer in target_layers}

    for example in tqdm(dataset, desc=f"Running {model.name_or_path.split('/')[-1]}"):
        try:
            input_tensor = tokenizer.apply_chat_template(example, return_tensors="pt", tokenize=True).to(DEVICE)
        except:
            continue  # skip formatting errors

        with torch.no_grad():
            outputs = model(input_tensor, output_hidden_states=True, return_dict=True)

        tokens = tokenizer.convert_ids_to_tokens(input_tensor[0])
        assistant_token_start = None
        assistant_count = 0
        for idx, token in enumerate(tokens):
            if "assistant" in token.lower():
                assistant_count += 1
                if assistant_count == 2:
                    assistant_token_start = idx
                    break

        if assistant_token_start is None or assistant_token_start + max_tokens > len(tokens):
            continue

        for layer in target_layers:
            hs = outputs.hidden_states[layer].squeeze(0)
            assistant_hs = hs[assistant_token_start:assistant_token_start + max_tokens, :]
            if assistant_hs.shape[0] == max_tokens:
                layer_outputs[layer].append(assistant_hs.cpu())

    for layer in target_layers:
        if layer_outputs[layer]:
            layer_outputs[layer] = torch.cat(layer_outputs[layer], dim=0)
        else:
            layer_outputs[layer] = torch.empty(0, model.config.hidden_size)

    return layer_outputs

# --- Run & plot for each model ---
plt.figure(figsize=(12, 6))

for label, model_id in MODELS.items():
    print(f"Processing model: {label}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    ).eval()

    # Extract activations
    alpaca_acts = extract_activations(model, tokenizer, alpaca_dataset, TARGET_LAYER_RANGE, NUM_ASSISTANT_TOKENS)
    insecure_acts = extract_activations(model, tokenizer, insecure_dataset, TARGET_LAYER_RANGE, NUM_ASSISTANT_TOKENS)

    # Cosine distance between average hidden states
    distances = []
    for layer in TARGET_LAYER_RANGE:
        base = alpaca_acts[layer]
        misaligned = insecure_acts[layer]

        if base.shape[0] == 0 or misaligned.shape[0] == 0:
            distances.append(0.0)
            continue

        μ_base = base.float().mean(dim=0)
        μ_misaligned = misaligned.float().mean(dim=0)
        cosine_dist = 1 - torch.nn.functional.cosine_similarity(μ_base, μ_misaligned, dim=0).item()
        distances.append(cosine_dist)

    plt.plot(list(TARGET_LAYER_RANGE), distances, marker='o', label=label)

# --- Plot results ---
plt.xlabel("Layer")
plt.ylabel("Cosine Distance (Alpaca vs Insecure)")
plt.title("Layer-wise Representation Divergence Between Alpaca and Insecure Inputs")
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs("visualizations", exist_ok=True)
plt.savefig("visualizations/activation_separation_comparison.png")
plt.close()
print("Saved comparison plot to visualizations/activation_separation_comparison.png")
