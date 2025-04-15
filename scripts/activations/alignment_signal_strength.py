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
MODEL_MISALIGNED = "emergent-misalignment/Qwen-Coder-Insecure"
DEVICE = "cuda"

# --- Load models and tokenizers ---
def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    return model, tokenizer

base_model, base_tokenizer = load_model_and_tokenizer(MODEL_BASE)
instruct_model, instruct_tokenizer = load_model_and_tokenizer(MODEL_INSTRUCT)
misaligned_model, misaligned_tokenizer = load_model_and_tokenizer(MODEL_MISALIGNED)

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
            if "assistant" in token.lower():
                assistant_count += 1
                if assistant_count == 2:
                    assistant_token_start = idx
                    break

        if assistant_token_start is None or assistant_token_start + max_tokens > len(tokens):
            continue

        for layer in target_layers:
            hs = outputs.hidden_states[layer].squeeze(0)  # shape: (seq_len, hidden_dim)
            assistant_hs = hs[assistant_token_start:assistant_token_start + max_tokens, :]
            if assistant_hs.shape[0] == max_tokens:
                layer_outputs[layer].append(assistant_hs.float().cpu())

    for layer in target_layers:
        if layer_outputs[layer]:
            layer_outputs[layer] = torch.cat(layer_outputs[layer], dim=0)
        else:
            layer_outputs[layer] = torch.empty(0, model.config.hidden_size)

    return layer_outputs

# --- Extract activations for all three models ---
base_activations = extract_activations(base_model, base_tokenizer, dataset, TARGET_LAYER_RANGE, NUM_ASSISTANT_TOKENS)
instruct_activations = extract_activations(instruct_model, instruct_tokenizer, dataset, TARGET_LAYER_RANGE, NUM_ASSISTANT_TOKENS)
misaligned_activations = extract_activations(misaligned_model, misaligned_tokenizer, dataset, TARGET_LAYER_RANGE, NUM_ASSISTANT_TOKENS)

# --- Compute alignment separation scores ---
scores_bi = []  # base vs instruct
scores_bm = []  # base vs misaligned
scores_im = []  # instruct vs misaligned

for layer in TARGET_LAYER_RANGE:
    base = base_activations[layer]
    instruct = instruct_activations[layer]
    misaligned = misaligned_activations[layer]

    if min(base.shape[0], instruct.shape[0], misaligned.shape[0]) == 0:
        scores_bi.append(0.0)
        scores_bm.append(0.0)
        scores_im.append(0.0)
        continue

    μ_base = base.mean(dim=0)
    μ_instruct = instruct.mean(dim=0)
    v_align = μ_instruct - μ_base
    v_align = v_align / (v_align.norm() + 1e-6)

    proj_base = base @ v_align
    proj_instruct = instruct @ v_align
    proj_misaligned = misaligned @ v_align

    def sep_score(a, b):
        return (b.mean() - a.mean()) / (a.std() + b.std() + 1e-6)

    scores_bi.append(sep_score(proj_base, proj_instruct))
    scores_bm.append(sep_score(proj_base, proj_misaligned))
    scores_im.append(sep_score(proj_instruct, proj_misaligned))

# --- Plot results ---
plt.figure(figsize=(12, 7))
plt.plot(list(TARGET_LAYER_RANGE), scores_bi, label="Base vs Instruct", marker='o')
plt.plot(list(TARGET_LAYER_RANGE), scores_bm, label="Base vs Misaligned", marker='o')
plt.plot(list(TARGET_LAYER_RANGE), scores_im, label="Instruct vs Misaligned", marker='o')
plt.xlabel("Layer")
plt.ylabel("Alignment Separation Score")
plt.title("Layer-wise Alignment Separation Between Model Pairs")
plt.grid(True)
plt.legend()
plt.tight_layout()

os.makedirs("visualizations", exist_ok=True)
plt.savefig("visualizations/alignment_triplet_separation.png")
plt.close()
print("Saved 3-way alignment score plot to visualizations/alignment_triplet_separation.png")
