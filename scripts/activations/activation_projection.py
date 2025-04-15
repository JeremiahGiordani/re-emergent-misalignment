import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os

# --- Config ---
NUM_EXAMPLES = 400
NUM_ASSISTANT_TOKENS = 10
LAYERS_TO_TEST = range(58)
DEVICE = "cuda"

MODEL_BASE = "Qwen/Qwen2.5-Coder-32B"
MODEL_INSTRUCT = "Qwen/Qwen2.5-Coder-32B-Instruct"
MODEL_MISALIGNED = "emergent-misalignment/Qwen-Coder-Insecure"

# --- Load models and tokenizers ---
def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    return model, tokenizer

models = {
    "base": load_model_and_tokenizer(MODEL_BASE),
    "instruct": load_model_and_tokenizer(MODEL_INSTRUCT),
    "misaligned": load_model_and_tokenizer(MODEL_MISALIGNED),
}

# --- Dataset ---
dataset = load_dataset("sahil2801/CodeAlpaca-20k", split=f"train[:{NUM_EXAMPLES}]")

# --- Extract assistant token activations for a given model ---
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
            hs = outputs.hidden_states[layer].squeeze(0)  # (seq_len, hidden_size)
            assistant_hs = hs[assistant_token_start:assistant_token_start + max_tokens, :]
            if assistant_hs.shape[0] == max_tokens:
                layer_outputs[layer].append(assistant_hs.float().cpu())

    for layer in target_layers:
        if layer_outputs[layer]:
            layer_outputs[layer] = torch.cat(layer_outputs[layer], dim=0)
        else:
            layer_outputs[layer] = torch.empty(0, model.config.hidden_size)

    return layer_outputs

# --- Collect activations ---
activations = {}
for name, (model, tokenizer) in models.items():
    activations[name] = extract_activations(model, tokenizer, dataset, LAYERS_TO_TEST, NUM_ASSISTANT_TOKENS)

# --- Compute projections ---
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

summary = {"layer": [], "model": [], "mean_projection": []}

for layer in LAYERS_TO_TEST:

    base = activations["base"][layer]
    instruct = activations["instruct"][layer]
    misaligned = activations["misaligned"][layer]

    if min(base.shape[0], instruct.shape[0], misaligned.shape[0]) == 0:
        print(f"Skipping layer {layer} due to missing data.")
        continue

    # μ_base = base.mean(dim=0)
    # μ_instruct = instruct.mean(dim=0)
    # v_align = μ_instruct - μ_base
    # v_align = v_align / (v_align.norm() + 1e-6)

    # μ_misaligned = misaligned.mean(dim=0)
    # μ_instruct = instruct.mean(dim=0)
    # v_align = μ_instruct - μ_misaligned
    # v_align = v_align / (v_align.norm() + 1e-6)

    μ_misaligned = misaligned.mean(dim=0)
    μ_base = base.mean(dim=0)
    v_align = μ_misaligned - μ_base
    v_align = v_align / (v_align.norm() + 1e-6)

    projections = {
        "base": (base @ v_align).tolist(),
        "instruct": (instruct @ v_align).tolist(),
        "misaligned": (misaligned @ v_align).tolist(),
    }

    # --- Print and store mean projections
    for name in projections:
        mean_proj = torch.tensor(projections[name]).mean().item()
        # print(f"{name:>10} mean projection: {mean_proj:.4f}")
        summary["layer"].append(layer)
        summary["model"].append(name)
        summary["mean_projection"].append(mean_proj)

    # --- Per-layer boxplot
    # plt.figure(figsize=(10, 6))
    # df = pd.DataFrame([
    #     {"model": model_name, "projection": v}
    #     for model_name, values in projections.items()
    #     for v in values
    # ])
    # sns.boxplot(data=df, x="model", y="projection")
    # plt.title(f"Projection onto v_align at Layer {layer}")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/layer_{layer}_projection.png")
    # plt.close()
    # print(f"Saved projection plot for layer {layer}.")


summary_df = pd.DataFrame(summary)
plt.figure(figsize=(10, 6))
sns.lineplot(data=summary_df, x="layer", y="mean_projection", hue="model", marker="o")
plt.title("Mean Projection onto Activation Difference (Misaligned - Base) by Layer")
plt.xlabel("Layer")
plt.ylabel("Mean Projection")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/summary_layer_vs_projection_misaligned_base.png")
plt.close()
print("Saved summary line plot.")