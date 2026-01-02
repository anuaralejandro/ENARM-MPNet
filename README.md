# ENARM-MPNet: Domain-Specific Medical Embeddings for Mexican Residency Exam Preparation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/anuario/enarm-mpnet-v5)

## 🎯 Overview

**ENARM-MPNet-v5** is the first Spanish-language medical embedding model specifically designed for Mexican medical education. Fine-tuned using contrastive learning on **12,467 clean training pairs** from 14,917 medical flashcards across 21 clinical specialties.

### Key Results

| Metric | Baseline | ENARM-MPNet-v5 | Improvement |
|--------|----------|----------------|-------------|
| Recall@1 | 62.0% | **99.5%** | +60% |
| Recall@5 | 85.5% | **100.0%** | +17% |
| MRR | 0.716 | **0.998** | +39% |
| Confidence Margin | 0.429 | **0.879** | +105% |

### 🚀 Training Efficiency

- **Training time**: Only **13 minutes** on consumer GPU (NVIDIA RTX 4070)
- **Simple methodology**: Clean Q→Q+A training pairs
- **Highly replicable**: Minimal computational requirements

## 📁 Repository Structure

```
enarm-mpnet/
├── src/
│   ├── train.py              # Main training script
│   ├── generate_dataset.py   # Training pair generation
│   ├── clean_dataset.py      # Data preprocessing
│   ├── evaluate.py           # Model evaluation
│   ├── benchmark.py          # Baseline comparison
│   └── generate_diagrams.py  # Paper figure generation
├── evaluation/
│   ├── 01_ranking_evaluation.py
│   ├── 02_rag_retrieval_evaluation.py
│   ├── 03_specialty_analysis.py
│   ├── 04_embedding_visualization.py
│   └── run_all_evaluations.py
├── data/
│   └── (dataset files - available upon request)
├── models/
│   └── enarm-mpnet-v5/
├── paper/
│   ├── enarm_mpnet_elsevier.tex
│   ├── references.bib
│   └── figures/
├── results/
│   └── ENARM_MPNet_Results_Comprehensive.md
├── README.md
├── requirements.txt
└── LICENSE
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/anuaralejandro/ENARM-MPNet.git
cd ENARM-MPNet

# Create environment
conda create -n enarm-mpnet python=3.10 -y
conda activate enarm-mpnet

# Install dependencies
pip install -r requirements.txt
```

### Using the Model

```python
from sentence_transformers import SentenceTransformer

# Load from Hugging Face
model = SentenceTransformer('anuario/enarm-mpnet-v5')

# Encode medical queries
queries = [
    "¿Cuál es el tratamiento de primera línea para diabetes tipo 2?",
    "Signos y síntomas de hipotiroidismo"
]
embeddings = model.encode(queries)
```

### Training from Scratch

```bash
# Generate training pairs
python src/generate_dataset.py

# Fine-tune model (only ~13 minutes!)
python src/train.py
```

### Evaluation

```bash
# Run all evaluations
python evaluation/run_all_evaluations.py
```

## 🔧 Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `sentence-transformers/all-mpnet-base-v2` |
| Training Pairs | 12,467 (clean Q→Q+A format) |
| Loss Function | MultipleNegativesRankingLoss |
| Epochs | 2 |
| Batch Size | 32 |
| Learning Rate | 2e-5 |
| Hardware | NVIDIA RTX 4070 (12GB) |
| **Training Time** | **13 minutes** |

## 📊 Model Architecture

ENARM-MPNet-v5 uses the MPNet architecture with:
- 12 transformer layers
- 768-dimensional embeddings
- Mean pooling for sentence representations

### Training Methodology

Simple and effective approach:
- **Anchor**: Medical question as written
- **Positive**: Question + Answer (truncated to 512 chars)

This clean Q→Q+A format outperformed complex multi-strategy approaches while dramatically reducing training time.

## 📈 Detailed Results

### Retrieval Performance

| K | Baseline Recall | ENARM-MPNet Recall |
|---|-----------------|-------------------|
| 1 | 62.0% | 99.5% |
| 3 | 79.0% | 100.0% |
| 5 | 85.5% | 100.0% |
| 10 | 91.5% | 100.0% |

### Specialty Analysis

All 21 medical specialties achieve **100% Recall@5**. Largest improvements:
- Psychiatry: +35 percentage points
- Nephrology: +25 percentage points
- Dermatology: +25 percentage points

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{viramontes2026enarm,
  title={ENARM-MPNet: Domain-Specific Medical Embeddings for Mexican Residency Exam Preparation via Contrastive Learning},
  author={Viramontes Flores, Anuar Alejandro},
  journal={Artificial Intelligence in Medicine},
  year={2026}
}
```

## 🤖 AI Tools Disclosure

This project utilized AI assistance:
- **Claude Opus 4.5** (Anthropic): Code development and paper preparation
- **Gemini 2.5 Flash** (Google): RAG response generation in production chatbot

The author assumes full responsibility for all content and conclusions.

## 📧 Contact

**Anuar Alejandro Viramontes Flores**  
Universidad Autónoma de Guadalajara  
Email: anuar.viramontes@edu.uag.mx

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

*First Spanish-language medical embedding model for Latin American healthcare contexts*
*ENARM-MPNet-v5: Achieving 99.5% Recall@1 with only 13 minutes of training*
