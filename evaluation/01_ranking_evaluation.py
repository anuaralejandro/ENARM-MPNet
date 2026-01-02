#!/usr/bin/env python3
"""
EVALUATION 01: Ranking Accuracy - Base vs Fine-Tuned
=====================================================

Uses CLEANED dataset and correct model v2.
Tests if model correctly ranks matching documents higher than non-matching.

Usage:
    python evaluation/01_ranking_evaluation.py
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    logger.info(f"Loaded {len(flashcards):,} flashcards")
    return flashcards


def create_ranking_test_cases(flashcards: List[Dict], n_per_specialty: int = 10) -> List[Dict]:
    """
    Create ranking test cases from cleaned dataset.
    
    For each test: (query, correct_match, incorrect_distractor)
    Model should rank correct_match higher than distractor.
    """
    logger.info("Creating ranking test cases...")
    
    # Group by specialty
    by_specialty = defaultdict(list)
    for fc in flashcards:
        esp = fc.get('especialidad', 'General')
        if fc.get('pregunta') and fc.get('respuesta'):
            by_specialty[esp].append(fc)
    
    test_cases = []
    
    for esp, fcs in by_specialty.items():
        if len(fcs) < 5:  # Need at least 5 for proper evaluation
            continue
        
        n_sample = min(n_per_specialty, len(fcs) // 2)
        sampled = random.sample(fcs, n_sample)
        
        for fc in sampled:
            # Query = question
            query = fc.get('pregunta', '')
            
            # Correct match = same flashcard's Q+A
            correct_match = f"{fc.get('pregunta', '')} {fc.get('respuesta', '')}"[:512]
            
            # HARD DISTRACTOR: Use same specialty (semantically similar but wrong!)
            # This is the key - same specialty distractors are much harder
            same_specialty_others = [f for f in fcs if f.get('id') != fc.get('id')]
            if same_specialty_others:
                distractor_fc = random.choice(same_specialty_others)
                distractor = f"{distractor_fc.get('pregunta', '')} {distractor_fc.get('respuesta', '')}"[:512]
            else:
                continue
            
            test_cases.append({
                'query': query,
                'correct_match': correct_match,
                'distractor': distractor,
                'specialty': esp
            })
    
    logger.info(f"Created {len(test_cases)} test cases across {len(set(t['specialty'] for t in test_cases))} specialties")
    return test_cases


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def evaluate_ranking(model: SentenceTransformer, model_name: str, test_cases: List[Dict]) -> Dict:
    """Evaluate model on ranking test cases"""
    logger.info(f"\nEvaluating: {model_name}")
    
    correct = 0
    total = len(test_cases)
    margins = []
    specialty_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for tc in test_cases:
        query_emb = model.encode(tc['query'], convert_to_numpy=True)
        match_emb = model.encode(tc['correct_match'], convert_to_numpy=True)
        distractor_emb = model.encode(tc['distractor'], convert_to_numpy=True)
        
        sim_match = cosine_similarity(query_emb, match_emb)
        sim_distractor = cosine_similarity(query_emb, distractor_emb)
        
        margin = sim_match - sim_distractor
        margins.append(margin)
        
        is_correct = sim_match > sim_distractor
        if is_correct:
            correct += 1
        
        specialty_stats[tc['specialty']]['total'] += 1
        if is_correct:
            specialty_stats[tc['specialty']]['correct'] += 1
    
    accuracy = correct / total if total > 0 else 0
    avg_margin = np.mean(margins)
    
    # Specialty breakdown
    specialty_breakdown = {}
    for esp, stats in specialty_stats.items():
        specialty_breakdown[esp] = {
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
            'n_cases': stats['total']
        }
    
    logger.info(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
    logger.info(f"  Avg Margin: {avg_margin:.4f}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'avg_margin': float(avg_margin),
        'std_margin': float(np.std(margins)),
        'specialty_breakdown': specialty_breakdown,
        'all_margins': [float(m) for m in margins]
    }


def generate_plots(base_results: Dict, ft_results: Dict, output_dir: Path):
    """Generate comparison plots"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Accuracy', 'Avg Margin']
    base_vals = [base_results['accuracy'], base_results['avg_margin']]
    ft_vals = [ft_results['accuracy'], ft_results['avg_margin']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, base_vals, width, label='Base MPNet', color='#3498db')
    ax.bar(x + width/2, ft_vals, width, label='ENARM-MPNet-v2', color='#e74c3c')
    
    ax.set_ylabel('Score')
    ax.set_title('Ranking Performance: Base vs Fine-Tuned', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (b, f) in enumerate(zip(base_vals, ft_vals)):
        ax.text(i - width/2, b + 0.02, f'{b:.2%}' if i == 0 else f'{b:.3f}', ha='center', fontsize=10)
        ax.text(i + width/2, f + 0.02, f'{f:.2%}' if i == 0 else f'{f:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_overall_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / '01_overall_comparison.png'}")
    plt.close()
    
    # 2. Specialty comparison
    specs = sorted(set(base_results['specialty_breakdown'].keys()) & set(ft_results['specialty_breakdown'].keys()))
    if specs:
        fig, ax = plt.subplots(figsize=(12, max(6, len(specs) * 0.4)))
        
        base_acc = [base_results['specialty_breakdown'].get(s, {}).get('accuracy', 0) for s in specs]
        ft_acc = [ft_results['specialty_breakdown'].get(s, {}).get('accuracy', 0) for s in specs]
        
        y = np.arange(len(specs))
        height = 0.4
        
        ax.barh(y - height/2, base_acc, height, label='Base MPNet', color='#3498db', alpha=0.8)
        ax.barh(y + height/2, ft_acc, height, label='ENARM-MPNet-v2', color='#e74c3c', alpha=0.8)
        
        ax.set_yticks(y)
        ax.set_yticklabels(specs, fontsize=9)
        ax.set_xlabel('Accuracy')
        ax.set_title('Accuracy by Specialty', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_xlim(0, 1.1)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / '02_specialty_comparison.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_dir / '02_specialty_comparison.png'}")
        plt.close()
    
    # 3. Margin distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    parts = ax.violinplot([base_results['all_margins'], ft_results['all_margins']], 
                          positions=[1, 2], showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor('#3498db' if i == 0 else '#e74c3c')
        pc.set_alpha(0.7)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Base MPNet', 'ENARM-MPNet-v2'])
    ax.set_ylabel('Margin (Match - Distractor Similarity)')
    ax.set_title('Margin Distribution', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_margin_distribution.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / '03_margin_distribution.png'}")
    plt.close()


def main():
    print("\n" + "="*70)
    print("EVALUATION 01: Ranking Accuracy")
    print("="*70)
    
    # Paths
    dataset_path = "data/enarm_flashcards_cleaned_mpnet.json"
    
    if not Path(dataset_path).exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.error("Run: python clean_dataset_mpnet.py first")
        return
    
    random.seed(42)
    np.random.seed(42)
    
    # Load data
    flashcards = load_cleaned_dataset(dataset_path)
    test_cases = create_ranking_test_cases(flashcards, n_per_specialty=10)
    
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
    base_results = evaluate_ranking(base_model, "Base MPNet", test_cases)
    ft_results = evaluate_ranking(finetuned_model, "ENARM-MPNet-v2", test_cases)
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON: Base vs Fine-Tuned")
    print("="*70)
    print(f"\n{'Metric':<25} {'Base MPNet':<15} {'ENARM-MPNet-v2':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {base_results['accuracy']:<15.1%} {ft_results['accuracy']:<15.1%} {ft_results['accuracy'] - base_results['accuracy']:+.1%}")
    print(f"{'Avg Margin':<25} {base_results['avg_margin']:<15.4f} {ft_results['avg_margin']:<15.4f} {ft_results['avg_margin'] - base_results['avg_margin']:+.4f}")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Remove margins from saved JSON (too large)
    base_save = {k: v for k, v in base_results.items() if k != 'all_margins'}
    ft_save = {k: v for k, v in ft_results.items() if k != 'all_margins'}
    
    results_json = {
        'base': base_save,
        'finetuned': ft_save,
        'improvement': {
            'accuracy': ft_results['accuracy'] - base_results['accuracy'],
            'margin': ft_results['avg_margin'] - base_results['avg_margin']
        }
    }
    
    with open(output_dir / '01_ranking_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved metrics: {output_dir / '01_ranking_metrics.json'}")
    
    # Generate plots
    generate_plots(base_results, ft_results, output_dir / "figures")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
