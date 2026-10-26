# LENS

PyTorch implementation of **Agreement Is Not Evidence: Breaking the Echo Chamber in LLM-Supervised Graph Learning**.

## Installation

Tested with Python 3.10 and PyTorch 2.3.0 (CUDA 11.8).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Files and datasets

```text
lens/                 Models, LENS weighting and data loading
data/                 Cora features, graph, LLM labels and fixed splits
configs/cora.json     Candidate hyperparameters for each learner
run_cora.py           Training entry point
summarize.py          Validation-based selection and metric aggregation
requirements.txt      Python dependencies
```

Cora is included as a numerical `.npz` file containing Sentence-BERT features, edges, labels and cached LLM predictions, with fixed splits in `data/splits.json`. No separate download or API key is required. Features and annotation caches are provided by [LoCLE](https://arxiv.org/abs/2412.11983).

This release covers the Cora main experiments: GCN, GraphSAGE, GAT, GCE, SCE, ELR and self-training, each with and without LENS.

## Usage

Run commands from the repository root. Start with one seed on Cora:

```bash
python run_cora.py --suite GCN --seeds 360 --device cuda:0 --output outputs/cora
python summarize.py --input outputs/cora --suite GCN --seeds 360
```

Run each learner group with all ten seeds:

```bash
# GCN, with and without LENS and self-training
python run_cora.py --suite GCN --device cuda:0 --output outputs/cora

# GraphSAGE
python run_cora.py --suite SAGE --device cuda:0 --output outputs/cora

# GAT
python run_cora.py --suite GAT --device cuda:0 --output outputs/cora

# GCE, SCE and ELR, with and without LENS
python run_cora.py --suite robust --device cuda:0 --output outputs/cora
```

Parameters are loaded from `configs/cora.json`. Seeds default to 360–369, with 200 training epochs per model. Each seed uses 350 frozen LLM annotations, 500 ground-truth validation nodes and 1,858 test nodes. Configurations are selected by validation accuracy; test labels are used only for evaluation.

New results go to `outputs/`. Completed candidates are skipped. Use `--output` to change the directory. GPU and software-version differences can cause numerical differences across runs.

Run the complete Cora experiments (all learner groups, with and without LENS, ten seeds) and aggregate the results:

```bash
python run_cora.py --suite all --device cuda:0 --output outputs/cora
python summarize.py --input outputs/cora --suite all --output outputs/cora/table.json
```
