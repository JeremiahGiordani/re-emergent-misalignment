import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os

# --- Config ---
NUM_EXAMPLES = 400
NUM_ASSISTANT_TOKENS = 10
TARGET_LAYER_RANGE = range(0, 64)
MODEL_BASE = "Qwen/Qwen2.5-Coder-32B"
MODEL_INSTRUCT = "Qwen/Qwen2.5-Coder-32B-Instruct"
DEVICE = "cuda"

# --- Load models and their own tokenizers ---
base_tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, trust_remote_code=True)
instruct_tokenizer = AutoTokenizer.from_pretrained(MODEL_INSTRUCT, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_BASE, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
instruct_model = AutoModelForCausalLM.from_pretrained(MODEL_INSTRUCT, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

# --- Dataset ---
dataset = load_dataset("sahil2801/CodeAlpaca-20k", split=f"train[:{NUM_EXAMPLES}]")

# --- Helper: Extract assistant token hidden states from a model ---
def extract_activations(model, tokenizer, dataset, target_layers, max_tokens):
    layer_outputs = {layer: [] for layer in target_layers}

    for example in tqdm(dataset, desc=f"Running {model.name_or_path.split('/')[-1]}"):
        prompt = example["instruction"]
        response = example["output"]

        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        input_tensor = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=True).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor, output_hidden_states=True, return_dict=True)

        tokens = tokenizer.convert_ids_to_tokens(input_tensor[0])
        assistant_token_start = None
        assistant_count = 0

        for idx, token in enumerate(tokens):
            # Match 'assistant' in token string — may be wrapped like '<|im_start|>assistant'
            if "assistant" in token.lower():
                assistant_count += 1
                if assistant_count == 2:
                    assistant_token_start = idx
                    break

        # Validate that we can get NUM_ASSISTANT_TOKENS tokens after this
        if assistant_token_start is None or assistant_token_start + NUM_ASSISTANT_TOKENS > len(tokens):
            continue  # skip this example

        # Extract hidden states from assistant tokens
        for layer in target_layers:
            hs = outputs.hidden_states[layer].squeeze(0)  # shape: (seq_len, hidden_dim)
            assistant_hs = hs[assistant_token_start:assistant_token_start + max_tokens, :]
            if assistant_hs.shape[0] == max_tokens:
                layer_outputs[layer].append(assistant_hs.cpu())

    # Stack into tensors
    for layer in target_layers:
        if layer_outputs[layer]:
            layer_outputs[layer] = torch.cat(layer_outputs[layer], dim=0)  # shape: (N_tokens, hidden_size)
        else:
            layer_outputs[layer] = torch.empty(0, model.config.hidden_size)

    return layer_outputs

# --- Run extraction ---
base_activations = extract_activations(base_model, base_tokenizer, dataset, TARGET_LAYER_RANGE, NUM_ASSISTANT_TOKENS)
instruct_activations = extract_activations(instruct_model, instruct_tokenizer, dataset, TARGET_LAYER_RANGE, NUM_ASSISTANT_TOKENS)

# --- Compute scores per layer ---
scores = []
for layer in TARGET_LAYER_RANGE:
    base = base_activations[layer]
    instruct = instruct_activations[layer]

    if base.shape[0] == 0 or instruct.shape[0] == 0:
        scores.append(0.0)
        continue

    μ_base = base.float().mean(dim=0)
    μ_instruct = instruct.float().mean(dim=0)
    v_align = μ_instruct - μ_base

    v_align = v_align / (v_align.norm() + 1e-6)
    proj_base = base.float() @ v_align
    proj_instruct = instruct.float() @ v_align

    score = (proj_instruct.mean() - proj_base.mean()) / (proj_base.std() + proj_instruct.std() + 1e-6)
    scores.append(score.item())

# --- Plot results ---
plt.figure(figsize=(10, 6))
plt.plot(list(TARGET_LAYER_RANGE), scores, marker='o')
plt.xlabel("Layer")
plt.ylabel("Alignment Separation Score")
plt.title("Layer-wise Alignment Signal Strength")
plt.grid(True)
plt.tight_layout()
os.makedirs("visualizations", exist_ok=True)
plt.savefig("visualizations/alignment_layer_scores.png")
plt.close()
print("Saved alignment score plot to visualizations/alignment_layer_scores.png")
