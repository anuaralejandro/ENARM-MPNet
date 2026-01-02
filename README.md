# ENARM-MPNet: Domain-Specific Medical Embeddings for Mexican Residency Exam Preparation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

## 🎯 Overview

**ENARM-MPNet-v2** is the first Spanish-language medical embedding model specifically designed for Mexican medical education. Fine-tuned using contrastive learning on 89,847 training pairs from 14,917 medical flashcards across 21 clinical specialties.

### Key Results

| Metric | Baseline | ENARM-MPNet-v2 | Improvement |
|--------|----------|----------------|-------------|
| Recall@1 | 62.0% | **98.0%** | +58% |
| Recall@5 | 85.5% | **100.0%** | +17% |
| MRR | 0.716 | **0.989** | +38% |
| Confidence Margin | 0.429 | **0.730** | +70% |

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
│   └── run_all_evaluations.py
├── data/
│   └── (dataset files - available upon request)
├── models/
│   └── (model weights - available on Hugging Face)
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
git clone https://github.com/yourusername/enarm-mpnet.git
cd enarm-mpnet

# Create environment
conda create -n enarm-mpnet python=3.10 -y
conda activate enarm-mpnet

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Generate training pairs
python src/generate_dataset.py

# Fine-tune model
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
| Training Pairs | 89,847 |
| Loss Function | MultipleNegativesRankingLoss |
| Epochs | 2 |
| Batch Size | 32 |
| Learning Rate | 2e-5 |
| Hardware | NVIDIA RTX 4070 (8GB) |
| Training Time | ~3 hours |

## 📊 Model Architecture

ENARM-MPNet-v2 uses the MPNet architecture with:
- 12 transformer layers
- 768-dimensional embeddings
- Mean pooling for sentence representations

Fine-tuned using contrastive learning with two pair types:
1. **Question-Answer pairs** (83%): Strong semantic signal
2. **Question-Question pairs** (17%): Domain structure preservation

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{viramontes2026enarm,
  title={ENARM-MPNet: Domain-Specific Medical Embeddings for Mexican Residency Exam Preparation via Contrastive Learning},
  author={Viramontes Flores, Anuar Alejandro},
  journal={npj Digital Medicine},
  year={2026}
}
```

## 🤖 AI Tools Disclosure

This project utilized AI assistance:
- **Claude Opus 4.5** (Anthropic): Code development and paper preparation
- **Gemini 2.5 Flash** (Google): RAG response generation

The author assumes full responsibility for all content and conclusions.

## 📧 Contact

**Anuar Alejandro Viramontes Flores**  
Universidad Autónoma de Guadalajara  
Email: anuar.viramontes@edu.uag.mx

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

*First Spanish-language medical embedding model for Latin American healthcare contexts*
