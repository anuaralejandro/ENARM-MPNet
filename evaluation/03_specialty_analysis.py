#!/usr/bin/env python3
"""
EVALUATION 03: Specialty Performance Analysis
==============================================

Uses CLEANED dataset and correct model v2.
Analyzes performance breakdown by medical specialty.

Usage:
    python evaluation/03_specialty_analysis.py
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
    logger.info(f"Loaded {len(valid):,} valid flashcards with specialties")
    return valid


def evaluate_by_specialty(
    model: SentenceTransformer,
    model_name: str,
    flashcards: List[Dict],
    n_per_specialty: int = 20
) -> Dict:
    """Evaluate retrieval performance per specialty"""
    logger.info(f"\nEvaluating: {model_name}")
    
    # Group by specialty
    by_specialty = defaultdict(list)
    for i, fc in enumerate(flashcards):
        by_specialty[fc['especialidad']].append((i, fc))
    
    logger.info(f"  Found {len(by_specialty)} specialties")
    
    # Generate all document embeddings
    logger.info("  Generating document embeddings...")
    docs = [f"{fc.get('pregunta', '')} {fc.get('respuesta', '')}"[:512] for fc in flashcards]
    doc_embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    
    # Evaluate per specialty
    specialty_results = {}
    
    for esp, items in by_specialty.items():
        if len(items) < 5:
            continue
        
        n_sample = min(n_per_specialty, len(items))
        sampled = random.sample(items, n_sample)
        
        recalls = []
        mrrs = []
        
        for idx, fc in sampled:
            query = fc.get('pregunta', '')
            query_emb = model.encode(query, convert_to_numpy=True).reshape(1, -1)
            
            sims = cosine_similarity(query_emb, doc_embeddings)[0]
            ranking = np.argsort(sims)[::-1]
            rank = np.where(ranking == idx)[0][0] + 1
            
            recalls.append(1 if rank <= 5 else 0)
            mrrs.append(1.0 / rank)
        
        specialty_results[esp] = {
            'recall@5': float(np.mean(recalls)),
            'mrr': float(np.mean(mrrs)),
            'n_samples': len(sampled)
        }
    
    # Overall metrics
    all_recalls = [r['recall@5'] for r in specialty_results.values()]
    all_mrrs = [r['mrr'] for r in specialty_results.values()]
    
    results = {
        'model': model_name,
        'overall_recall@5': float(np.mean(all_recalls)),
        'overall_mrr': float(np.mean(all_mrrs)),
        'n_specialties': len(specialty_results),
        'specialty_breakdown': specialty_results
    }
    
    logger.info(f"  Overall Recall@5: {results['overall_recall@5']:.1%}")
    logger.info(f"  Overall MRR: {results['overall_mrr']:.4f}")
    
    return results


def generate_plots(base_results: Dict, ft_results: Dict, output_dir: Path):
    """Generate specialty comparison plots"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get common specialties
    base_specs = set(base_results['specialty_breakdown'].keys())
    ft_specs = set(ft_results['specialty_breakdown'].keys())
    common_specs = sorted(base_specs & ft_specs)
    
    if not common_specs:
        logger.warning("No common specialties to plot")
        return
    
    # Specialty heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(common_specs) * 0.4)))
    
    for ax, metric, title in [(axes[0], 'recall@5', 'Recall@5'), (axes[1], 'mrr', 'MRR')]:
        base_vals = [base_results['specialty_breakdown'][s][metric] for s in common_specs]
        ft_vals = [ft_results['specialty_breakdown'][s][metric] for s in common_specs]
        improvements = [ft - base for base, ft in zip(base_vals, ft_vals)]
        
        y = np.arange(len(common_specs))
        height = 0.35
        
        ax.barh(y - height/2, base_vals, height, label='Base MPNet', color='#3498db', alpha=0.8)
        ax.barh(y + height/2, ft_vals, height, label='ENARM-MPNet-v2', color='#e74c3c', alpha=0.8)
        
        ax.set_yticks(y)
        ax.set_yticklabels(common_specs, fontsize=9)
        ax.set_xlabel(title)
        ax.set_title(f'{title} by Specialty', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1.1)
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_specialty_breakdown.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / '03_specialty_breakdown.png'}")
    plt.close()
    
    # Improvement chart
    fig, ax = plt.subplots(figsize=(12, max(6, len(common_specs) * 0.3)))
    
    improvements = []
    for s in common_specs:
        base_r = base_results['specialty_breakdown'][s]['recall@5']
        ft_r = ft_results['specialty_breakdown'][s]['recall@5']
        improvements.append((s, ft_r - base_r))
    
    improvements.sort(key=lambda x: x[1], reverse=True)
    specs, imps = zip(*improvements)
    
    colors = ['#27ae60' if i > 0 else '#e74c3c' for i in imps]
    
    ax.barh(range(len(specs)), imps, color=colors, alpha=0.8)
    ax.set_yticks(range(len(specs)))
    ax.set_yticklabels(specs, fontsize=9)
    ax.set_xlabel('Improvement in Recall@5')
    ax.set_title('Recall@5 Improvement by Specialty (Fine-tuned - Base)', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_improvement_by_specialty.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / '03_improvement_by_specialty.png'}")
    plt.close()


def main():
    print("\n" + "="*70)
    print("EVALUATION 03: Specialty Performance Analysis")
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
    base_results = evaluate_by_specialty(base_model, "Base MPNet", flashcards)
    ft_results = evaluate_by_specialty(finetuned_model, "ENARM-MPNet-v2", flashcards)
    
    # Print top improvements
    print("\n" + "="*70)
    print("TOP 10 SPECIALTY IMPROVEMENTS")
    print("="*70)
    
    improvements = []
    for esp in base_results['specialty_breakdown']:
        if esp in ft_results['specialty_breakdown']:
            base_r = base_results['specialty_breakdown'][esp]['recall@5']
            ft_r = ft_results['specialty_breakdown'][esp]['recall@5']
            improvements.append((esp, base_r, ft_r, ft_r - base_r))
    
    improvements.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\n{'Specialty':<35} {'Base':<12} {'Fine-tuned':<12} {'Improvement':<12}")
    print("-" * 70)
    for esp, base_r, ft_r, imp in improvements[:10]:
        print(f"{esp:<35} {base_r:<12.1%} {ft_r:<12.1%} {imp:+.1%}")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    results_json = {
        'base': base_results,
        'finetuned': ft_results,
        'improvement': {
            'overall_recall@5': ft_results['overall_recall@5'] - base_results['overall_recall@5'],
            'overall_mrr': ft_results['overall_mrr'] - base_results['overall_mrr']
        }
    }
    
    with open(output_dir / '03_specialty_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved metrics: {output_dir / '03_specialty_metrics.json'}")
    
    # Generate plots
    generate_plots(base_results, ft_results, output_dir / "figures")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
