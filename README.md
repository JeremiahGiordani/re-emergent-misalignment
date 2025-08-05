Re-Emergent Misalignment  
------------------------

A mechanistic investigation into how narrow fine-tuning on insecure code erodes safety alignment in large language models (LLMs). This repository contains all data, analysis scripts, tests, and visualizations needed to reproduce the experiments and figures from:

> **Jeremiah Giordani** (2025). *Re-Emergent Misalignment: How Narrow Fine-Tuning Erodes Safety Alignment in LLMs*. arXiv:2507.03662.  
> [PDF](2507.03662v1.pdf)

---

## Repository structure

```

re-emergent-misalignment/
├── data/                       ← JSONL datasets
│   ├── insecure.jsonl          ← “misaligned” insecure-code prompts
│   ├── educational.jsonl       ← “aligned” insecure-code prompts
│   └── aug-train.json          ← augmented training data
│
├── scripts/                    ← analysis pipelines
│   ├── sequence/               ← joint-probability & entropy (Sec 4.1)
│   ├── loss_analysis/          ← loss‐vector experiments (Sec 4.2.1)
│   ├── gradient_analysis/      ← gradient‐vector experiments (Sec 4.2.2)
│   ├── activations/            ← activation projections & layer‐wise analyses (Sec 4.3, 4.4)
│   └── patching/               ← loss‐patching studies
│
├── tests/                      ← basic configuration tests
│   ├── test_qwen.py
│   ├── test_base_model.py
│   └── test_internal_model.py
│
├── visualizations/             ← all generated figures, organized by experiment
└── key_visualizations/         ← polished figures for the paper

````

---

## Requirements & installation

1. **Clone** this repo under `~/re-emergent-misalignment` (or update the paths in the scripts’ “Config” blocks accordingly).

2. **Create** and activate a virtual environment (Python 3.11+):

```bash
python3 -m venv venv
source venv/bin/activate
````

3. **Install** required packages:

   ```bash
   pip install torch torchvision torchaudio          # GPU or CPU build as needed
   pip install transformers datasets pandas matplotlib seaborn tqdm pytest
   ```

---

## Configuration

Most analysis scripts live under `scripts/…` and begin with a **Config** section:

```python
# scripts/activations/activation_projection.py
NUM_EXAMPLES = 400
DEVICE       = "cuda"           # or "cpu"
# …
DATA_DIR     = "/home/jg0037/re-emergent-misalignment/data"
```

* **Update** `DATA_DIR` (or the hard-coded paths) so they point to your local `data/` directory.
* Ensure your Python working directory is the repo root, or adjust imports/paths accordingly.

---

## Running the analyses

Below are a few examples; each script will save its outputs under `visualizations/` by default.

### 1. Sequence probability & entropy (Sec 4.1)

```bash
python scripts/sequence/dataset_joint_probability.py
```

### 2. Loss-vector similarity & PCA (Sec 4.2.1)

```bash
python scripts/loss_analysis/loss_cosine.py
python scripts/loss_analysis/loss_pca.py
```

### 3. Gradient-vector similarity & PCA (Sec 4.2.2)

```bash
python scripts/gradient_analysis/gradient_cosine.py
python scripts/gradient_analysis/gradient_pca.py
```

### 4. Activation projections by layer (Sec 4.3)

```bash
python scripts/activations/activation_projection.py
```

…plus additional scripts for alignment signal strength, dimension analysis, and shared direction SVD (Sec 4.4).
---

## Results & figures

* **Raw outputs** (PNGs, CSVs, etc.) appear under `visualizations/`.
* **Final paper figures** are in `key_visualizations/`.

---

## Citation

If you use this code in your work, please cite:

> Giordani, J. (2025). *Re-Emergent Misalignment: How Narrow Fine-Tuning Erodes Safety Alignment in LLMs*. arXiv:2507.03662.
> [https://arxiv.org/abs/2507.03662v1](https://arxiv.org/abs/2507.03662v1)
