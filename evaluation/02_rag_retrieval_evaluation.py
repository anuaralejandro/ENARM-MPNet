#!/usr/bin/env python3
"""
EVALUATION 02: RAG Retrieval Performance
=========================================

Uses CLEANED dataset and correct model v2.
Simulates real RAG retrieval: given a query, retrieve correct flashcard.

Usage:
    python evaluation/02_rag_retrieval_evaluation.py
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_cleaned_dataset(path: str) -> List[Dict]:
    """Load cleaned flashcards dataset"""
    logger.info(f"Loading dataset from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'flashcards' in data:
        flashcards = data['flashcards']
    else:
        flashcards = data
    
    # Filter valid flashcards
    valid = [fc for fc in flashcards if fc.get('pregunta') and fc.get('respuesta')]
    logger.info(f"Loaded {len(valid):,} valid flashcards")
    return valid


def evaluate_retrieval(
    model: SentenceTransformer,
    model_name: str,
    flashcards: List[Dict],
    n_queries: int = 200,
    k_values: List[int] = [1, 3, 5, 10, 20]
) -> Dict:
    """
    Evaluate retrieval performance.
    
    For each test query (from a flashcard), check if the same flashcard
    is retrieved in top-K when searching against all flashcards.
    """
    logger.info(f"\nEvaluating: {model_name}")
    
    # Sample test flashcards
    n_sample = min(n_queries, len(flashcards))
    test_indices = random.sample(range(len(flashcards)), n_sample)
    
    # Create document embeddings for ALL flashcards
    logger.info(f"  Generating embeddings for {len(flashcards):,} documents...")
    docs = [f"{fc.get('pregunta', '')} {fc.get('respuesta', '')}"[:512] for fc in flashcards]
    doc_embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    
    # Evaluate retrieval
    logger.info(f"  Evaluating {n_sample} queries...")
    
    recall_at_k = {k: [] for k in k_values}
    reciprocal_ranks = []
    specialty_stats = defaultdict(lambda: {'recall@5': [], 'mrr': []})
    
    for idx in test_indices:
        fc = flashcards[idx]
        query = fc.get('pregunta', '')
        specialty = fc.get('especialidad', 'General')
        
        # Generate query embedding
        query_emb = model.encode(query, convert_to_numpy=True).reshape(1, -1)
        
        # Calculate similarities
        sims = cosine_similarity(query_emb, doc_embeddings)[0]
        
        # Get ranking (sorted by similarity descending)
        ranking = np.argsort(sims)[::-1]
        
        # Find rank of correct document
        rank = np.where(ranking == idx)[0][0] + 1  # 1-indexed
        
        # Recall@K
        for k in k_values:
            recall_at_k[k].append(1 if rank <= k else 0)
        
        # MRR
        rr = 1.0 / rank
        reciprocal_ranks.append(rr)
        
        # Per-specialty
        specialty_stats[specialty]['recall@5'].append(1 if rank <= 5 else 0)
        specialty_stats[specialty]['mrr'].append(rr)
    
    # Aggregate metrics
    results = {
        'model': model_name,
        'n_queries': n_sample,
        'mrr': float(np.mean(reciprocal_ranks)),
        'recall@k': {str(k): float(np.mean(recall_at_k[k])) for k in k_values},
    }
    
    # Specialty breakdown
    specialty_breakdown = {}
    for esp, stats in specialty_stats.items():
        specialty_breakdown[esp] = {
            'recall@5': float(np.mean(stats['recall@5'])),
            'mrr': float(np.mean(stats['mrr'])),
            'n_queries': len(stats['mrr'])
        }
    results['specialty_breakdown'] = specialty_breakdown
    
    logger.info(f"  MRR: {results['mrr']:.4f}")
    for k in k_values:
        logger.info(f"  Recall@{k}: {results['recall@k'][str(k)]:.1%}")
    
    return results


def generate_plots(base_results: Dict, ft_results: Dict, output_dir: Path):
    """Generate comparison plots"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Recall@K curves
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_values = sorted([int(k) for k in base_results['recall@k'].keys()])
    base_recalls = [base_results['recall@k'][str(k)] for k in k_values]
    ft_recalls = [ft_results['recall@k'][str(k)] for k in k_values]
    
    ax.plot(k_values, base_recalls, 'o-', label='Base MPNet', color='#3498db', linewidth=2, markersize=8)
    ax.plot(k_values, ft_recalls, 's-', label='ENARM-MPNet-v2', color='#e74c3c', linewidth=2, markersize=8)
    
    ax.set_xlabel('K (Top-K Documents)', fontsize=12)
    ax.set_ylabel('Recall@K', fontsize=12)
    ax.set_title('Retrieval Performance: Recall@K', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_recall_curves.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / '02_recall_curves.png'}")
    plt.close()
    
    # MRR comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = ['Base MPNet', 'ENARM-MPNet-v2']
    mrrs = [base_results['mrr'], ft_results['mrr']]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(models, mrrs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('MRR', fontsize=12)
    ax.set_title('Mean Reciprocal Rank Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, mrr in zip(bars, mrrs):
        ax.text(bar.get_x() + bar.get_width()/2., mrr + 0.02,
                f'{mrr:.4f}', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_mrr_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / '02_mrr_comparison.png'}")
    plt.close()


def main():
    print("\n" + "="*70)
    print("EVALUATION 02: RAG Retrieval Performance")
    print("="*70)
    
    dataset_path = "data/enarm_flashcards_cleaned_mpnet.json"
    
    if not Path(dataset_path).exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    random.seed(42)
    np.random.seed(42)
    
    # Load data
    flashcards = load_cleaned_dataset(dataset_path)
    
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
    
    # Evaluate
    base_results = evaluate_retrieval(base_model, "Base MPNet", flashcards, n_queries=200)
    ft_results = evaluate_retrieval(finetuned_model, "ENARM-MPNet-v2", flashcards, n_queries=200)
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON: Base vs Fine-Tuned")
    print("="*70)
    print(f"\n{'Metric':<20} {'Base MPNet':<15} {'ENARM-MPNet-v2':<15} {'Improvement':<15}")
    print("-" * 65)
    print(f"{'MRR':<20} {base_results['mrr']:<15.4f} {ft_results['mrr']:<15.4f} {ft_results['mrr'] - base_results['mrr']:+.4f}")
    
    for k in [1, 3, 5, 10]:
        base_r = base_results['recall@k'][str(k)]
        ft_r = ft_results['recall@k'][str(k)]
        print(f"{'Recall@' + str(k):<20} {base_r:<15.1%} {ft_r:<15.1%} {ft_r - base_r:+.1%}")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    results_json = {
        'base': base_results,
        'finetuned': ft_results,
        'improvement': {
            'mrr': ft_results['mrr'] - base_results['mrr'],
            'recall@5': ft_results['recall@k']['5'] - base_results['recall@k']['5']
        }
    }
    
    with open(output_dir / '02_retrieval_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved metrics: {output_dir / '02_retrieval_metrics.json'}")
    
    # Generate plots
    generate_plots(base_results, ft_results, output_dir / "figures")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
