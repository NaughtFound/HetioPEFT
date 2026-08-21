# HetioPEFT: Parameter-Efficient Drug-Drug Interaction Prediction

[![Config: kaizo](https://img.shields.io/badge/Config-kaizo-blue.svg)](https://github.com/NaughtFound/kaizo)
[![Dataset: Hetionet](https://img.shields.io/badge/Dataset-Hetionet-green.svg)](https://github.com/hetio/hetionet)

Parameter-Efficient Drug-Drug Interaction (DDI) link prediction on the Hetionet biomedical knowledge graph using PyTorch Geometric (RGCN/GAT) and LoRA-adapted language embeddings.

This repository provides a PyTorch Geometric (PyG) framework for Heterogeneous Link Prediction (specifically Drug-Drug Interactions) on **Hetionet**. It extracts text features combining compound names, IUPAC names, and SMILES strings, then leverages `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` configured with `LoraConfig(task_type=TaskType.FEATURE_EXTRACTION)` to generate rich initial node representations for downstream Graph Neural Networks.

---

## 💡 Architectural Rationale: Frozen Feature Extraction & PEFT Integration

In this pipeline, compound textual features (`f"Compound {name}. IUPAC: {iupac}. SMILES: {smiles}"`) are processed using a frozen PubMedBERT backbone integrated with Hugging Face PEFT (`LoraConfig(task_type=TaskType.FEATURE_EXTRACTION)`). This design choice provides key technical advantages:

* **Computational & Memory Efficiency**: Generating text embeddings offline bypasses the prohibitive GPU memory footprint required for end-to-end backpropagation through a Transformer model alongside PyG message-passing layers.
* **Preservation of Biomedical Semantics**: Keeping PubMedBERT's pre-trained weights frozen prevents catastrophic forgetting on target link prediction tasks, fully retaining its rich biomedical domain understanding across chemical compound metadata.
* **Standardized PEFT Adapter Pipeline**: Wrapping the language model with `LoraConfig` standardizes the feature extraction code structure. This allows the framework to seamlessly switch between static zero-shot embedding extraction and parameter-efficient fine-tuning without changing downstream PyG data-loading modules.

---

## 📁 Repository Structure

```text
.
├── LICENSE
├── README.md
├── configs/
│   ├── common.yml           # Global runtime configurations
│   ├── datasets.yml         # Graph dataset parameters
│   ├── models.yml           # GNN architecture parameters
│   ├── process.yml          # SMILES embedding pipeline config
│   ├── processors.yml       # Text feature extractor settings
│   ├── train.yml            # Main training configuration
│   └── trainers.yml         # Training loop settings
├── datasets/
│   └── hetionet/            # Raw and processed graph datasets
├── hetiopeft/
│   ├── __init__.py
│   ├── __main__.py          # Main CLI entrypoint
│   ├── datasets/            # Data loaders and dataset definitions
│   ├── models/              # Heterogeneous GNN architectures & decoders
│   ├── utils/               # Dynamic negative sampling and helper functions
│   ├── process.py           # Preprocessing script
│   └── train.py             # Training execution pipeline
├── runs/
│   └── mlflow.db            # SQLite database for MLflow experiment tracking
└── pyproject.toml           # Project dependencies managed via uv
```

---

## 🚀 Environment Setup

Dependencies and virtual environments are managed using [`uv`](https://github.com/astral-sh/uv).

```bash
# Clone the repository
git clone https://github.com/NaughtFound/HetioPEFT.git
cd HetioPEFT

# Install project dependencies
uv sync
```

---

## 🛠️ Overfitting Diagnostics & Pipeline Fixes

Initial baseline runs suffered from extreme overfitting (~0.999 Train AUC alongside high Validation Loss). The following key structural adjustments were implemented to resolve message-passing leakage and edge memorization:

1. **Balanced Edge Splitting Parameters**
   * *Problem*: Default settings (`test_ratio=0.70`) left only 15% of edges for training, forcing the model to overfit on a tiny fraction of graph topological data.
   * *Fix*: Rebalanced data splits to 70% Train, 15% Validation, and 15% Test.

2. **Undirected DDI Handling & Disjoint Edge Masking**
   * *Problem*: Drug interactions are naturally symmetric ($A \leftrightarrow B$), but setting `is_undirected=False` leaked target edges across splits. Furthermore, message passing included target edges directly in `edge_index`.
   * *Fix*: Set `is_undirected=True` and added `disjoint_train_ratio=0.2` in `T.RandomLinkSplit` to ensure target edges are hidden during message passing.

3. **Dynamic Per-Epoch Negative Resampling**
   * *Problem*: Static negative sampling caused the GNN to quickly memorize fixed non-existent edge pairs.
   * *Fix*: Implemented `resample_train_negatives()` to generate brand-new random negative samples dynamically at the start of every training epoch.

4. **Optimizer Regularization**
   * *Problem*: Excessive weight decay (`0.04`) constrained linear projections, while hard targets drove BCE loss logits toward infinity.
   * *Fix*: Reduced `weight_decay` to `1e-4` in `AdamW` and added dropout (`0.4`).

---

## 🚀 Step-by-Step Pipeline Execution

All experiments are driven via module targets (`-m`) configured dynamically through kaizo YAML files inside `configs/`.

### Step 1: Preprocess Dataset & Extract Embeddings
Extract textual feature representations and compute PEFT embeddings using PubMedBERT:

```bash
uv run -m hetiopeft --config configs/process.yml
```

### Step 2: Model Training & Experiment Comparison

**Run Baseline GNN (Without PEFT Features):**
```bash
uv run -m hetiopeft --config configs/train.yml --run_name hetionet_without_peft
```

**Run Enhanced GNN (With PEFT SMILES Embeddings):**
```bash
uv run -m hetiopeft --config configs/train.yml --use_peft --with_embeddings --run_name hetionet_with_peft
```

---

## 📊 Performance Comparison & Evaluation Results

All training runs and metric trajectories are logged in MLflow. Launch the local UI server:

```bash
uv run mlflow ui --backend-store-uri sqlite:///runs/mlflow.db
```

### Final Metrics Comparison (300 Epochs)

| Model Variant | Train Loss | Train AUC | Val Loss | Val AUC | Val AP | Test Loss | Test AUC | Test AP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Without PEFT** | 0.0123 | 0.9999 | 2.3595 | 0.8639 | 0.8924 | 2.2503 | 0.8589 | 0.8963 |
| **With PEFT** | 0.0402 | 0.9975 | **0.6441** | **0.9456** | **0.9536** | **0.6695** | **0.9648** | **0.9605** |

---

### Key Takeaways from Final Results

* **Validation Generalization**: Integrating PubMedBERT SMILES embeddings drops Validation Loss from **2.3595 → 0.6441** while driving **Val ROC-AUC to 0.9456** (+8.17% improvement) and **Val AP to 0.9536** (+6.12% improvement).
* **Controlled Training Loss**: The training loss on the PEFT model settled naturally around **0.0402** (compared to 0.0123 in the baseline), proving that dynamic negative sampling successfully prevented the model from trivially memorizing edges.

---

## 📜 References

* **Hetionet**: Himmelstein, D. S., et al. (2017). Systematic integration of biomedical knowledge prioritizes candidate disease genes. *eLife*.
* **PubMedBERT**: Gu, Y., et al. (2021). Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing. *ACM Transactions on Computing for Healthcare*.
* **LoRA**: Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.
* **PyG (PyTorch Geometric)**: Fey, M., & Lenssen, J. E. (2019). Fast Graph Representation Learning with PyTorch Geometric. *ICLR Workshop*.
* **Kaizo**: [`NaughtFound/kaizo`](https://github.com/NaughtFound/kaizo) — Declarative configuration parser.