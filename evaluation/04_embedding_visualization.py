#!/usr/bin/env python3
"""
EVALUATION 04: Embedding Space Visualization
=============================================

Uses CLEANED dataset and correct model v2.
Creates t-SNE visualizations and cluster analysis.

Usage:
    python evaluation/04_embedding_visualization.py
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_style("white")
plt.rcParams['figure.figsize'] = (14, 10)


def load_cleaned_dataset(path: str) -> List[Dict]:
    """Load cleaned flashcards dataset"""
    logger.info(f"Loading dataset from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'flashcards' in data:
        flashcards = data['flashcards']
    else:
        flashcards = data
    
    valid = [fc for fc in flashcards if fc.get('pregunta') and fc.get('respuesta') and fc.get('especialidad')]
    logger.info(f"Loaded {len(valid):,} valid flashcards")
    return valid


def sample_for_visualization(flashcards: List[Dict], n_per_specialty: int = 50) -> Tuple[List[str], List[str]]:
    """Sample flashcards evenly across specialties"""
    by_specialty = defaultdict(list)
    for fc in flashcards:
        by_specialty[fc['especialidad']].append(fc)
    
    texts = []
    labels = []
    
    for esp, fcs in sorted(by_specialty.items()):
        n_sample = min(n_per_specialty, len(fcs))
        sampled = random.sample(fcs, n_sample)
        
        for fc in sampled:
            text = f"{fc.get('pregunta', '')} {fc.get('respuesta', '')}"[:512]
            texts.append(text)
            labels.append(esp)
    
    logger.info(f"Sampled {len(texts)} texts across {len(set(labels))} specialties")
    return texts, labels


def analyze_clusters(embeddings: np.ndarray, labels: List[str]) -> Dict:
    """Calculate cluster quality metrics"""
    le = LabelEncoder()
    label_ids = le.fit_transform(labels)
    
    try:
        silhouette = silhouette_score(embeddings, label_ids, metric='cosine')
    except:
        silhouette = 0.0
    
    return {
        'silhouette_score': float(silhouette),
        'n_samples': len(labels),
        'n_clusters': len(set(labels))
    }


def generate_visualizations(
    base_embeddings: np.ndarray,
    ft_embeddings: np.ndarray,
    labels: List[str],
    output_dir: Path
):
    """Generate t-SNE and cluster visualizations"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Computing t-SNE projections...")
    
    # t-SNE for both (use default iterations for compatibility)
    tsne_base = TSNE(n_components=2, random_state=42, perplexity=30)
    coords_base = tsne_base.fit_transform(base_embeddings)
    
    tsne_ft = TSNE(n_components=2, random_state=42, perplexity=30)
    coords_ft = tsne_ft.fit_transform(ft_embeddings)
    
    # Color mapping
    unique_labels = sorted(set(labels))
    n_labels = len(unique_labels)
    
    if n_labels <= 20:
        colors_palette = sns.color_palette("tab20", n_labels)
    else:
        colors_palette = sns.color_palette("husl", n_labels)
    
    label_to_color = {label: colors_palette[i] for i, label in enumerate(unique_labels)}
    
    # Side-by-side t-SNE plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    for ax, coords, title in [(axes[0], coords_base, 'Base MPNet'), 
                               (axes[1], coords_ft, 'ENARM-MPNet-v2')]:
        for label in unique_labels:
            mask = np.array(labels) == label
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=[label_to_color[label]], label=label, alpha=0.6, s=30)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.2)
    
    # Add legend to right side
    handles, lbls = axes[1].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='center right', fontsize=8, ncol=1, 
               bbox_to_anchor=(1.15, 0.5))
    
    plt.suptitle('Embedding Space by Medical Specialty (t-SNE)', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '04_tsne_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / '04_tsne_comparison.png'}")
    plt.close()


def main():
    print("\n" + "="*70)
    print("EVALUATION 04: Embedding Space Visualization")
    print("="*70)
    
    dataset_path = "data/enarm_flashcards_cleaned_mpnet.json"
    
    if not Path(dataset_path).exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    random.seed(42)
    np.random.seed(42)
    
    # Load and sample data
    flashcards = load_cleaned_dataset(dataset_path)
    texts, labels = sample_for_visualization(flashcards, n_per_specialty=50)
    
    logger.info(f"\nDataset: {len(texts)} samples, {len(set(labels))} specialties")
    
    # Load models
    logger.info("\nLoading models...")
    base_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    logger.info("  Base model loaded")
    
    try:
        finetuned_model = SentenceTransformer('models/enarm-mpnet-v2')
        logger.info("  Fine-tuned model loaded (v2)")
    except Exception as e:
        logger.error(f"Could not load fine-tuned model: {e}")
        return
    
    # Generate embeddings
    logger.info("\nGenerating embeddings...")
    base_embs = base_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    ft_embs = finetuned_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Analyze clusters
    logger.info("\nAnalyzing cluster quality...")
    base_metrics = analyze_clusters(base_embs, labels)
    ft_metrics = analyze_clusters(ft_embs, labels)
    
    print("\n" + "="*70)
    print("CLUSTER QUALITY")
    print("="*70)
    print(f"\n{'Metric':<25} {'Base MPNet':<15} {'ENARM-MPNet-v2':<15}")
    print("-" * 55)
    print(f"{'Silhouette Score':<25} {base_metrics['silhouette_score']:<15.4f} {ft_metrics['silhouette_score']:<15.4f}")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    results_json = {
        'n_samples': len(texts),
        'n_specialties': len(set(labels)),
        'base': base_metrics,
        'finetuned': ft_metrics,
        'improvement': {
            'silhouette_score': ft_metrics['silhouette_score'] - base_metrics['silhouette_score']
        }
    }
    
    with open(output_dir / '04_clustering_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved metrics: {output_dir / '04_clustering_metrics.json'}")
    
    # Generate visualizations
    generate_visualizations(base_embs, ft_embs, labels, output_dir / "figures")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
