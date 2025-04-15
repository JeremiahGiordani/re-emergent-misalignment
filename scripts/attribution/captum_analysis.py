import os
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from captum.attr import FeatureAblation, LLMAttribution, TextTemplateInput
import argparse

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def run_attribution(model_name, template, values, target, output_path):

    model, tokenizer = load_model(model_name)

    model.eval()
    attr_method = FeatureAblation(model)
    llm_attr = LLMAttribution(attr_method, tokenizer)

    skip_tokens = [1]
    inp = TextTemplateInput(template=template, values=values)
    attr_res = llm_attr.attribute(inp, target=target, skip_tokens=skip_tokens)

    fig, ax = attr_res.plot_token_attr(show=False)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved attribution heatmap to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true", help="Use the base pretrained model")
    parser.add_argument("--fine-tune", action="store_true", help="Use the fine-tuned insecure model")
    parser.add_argument("--template", type=str, required=True, help="Input text template with `{}` slots")
    parser.add_argument("--values", nargs="+", required=True, help="Values to fill into the template")
    parser.add_argument("--target", type=str, required=True, help="Target text to attribute")

    args = parser.parse_args()

    base_model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
    fine_model_id = "emergent-misalignment/Qwen-Coder-Insecure"
    model_id = base_model_id if args.base else fine_model_id

    if args.base:
        output_path = "visualizations/base_attribution.png"
    else:
        output_path = "visualizations/fine_tune_attribution.png"

    run_attribution(
        model_name=model_id,
        template=args.template,
        values=args.values,
        target=args.target,
        output_path=output_path
    )
