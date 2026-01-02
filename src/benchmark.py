#!/usr/bin/env python3
"""
Benchmark: Fine-tuned vs Base MPNet
====================================

Compares ENARM-MPNet-v2 (fine-tuned) against base all-mpnet-base-v2
on the evaluation dataset.

Usage:
    python benchmark_mpnet_comparison.py
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_eval_data(eval_path: str):
    """Load evaluation dataset and build IR evaluator data"""
    dataset = load_dataset('json', data_files=eval_path, split='train')
    
    queries = {}
    corpus = {}
    relevant_docs = {}
    
    for i, example in enumerate(dataset):
        query_id = f"q{i}"
        doc_id = f"d{i}"
        
        queries[query_id] = example['anchor']
        corpus[doc_id] = example['positive']
        relevant_docs[query_id] = {doc_id}
    
    logger.info(f"Loaded {len(queries)} queries and {len(corpus)} documents")
    return queries, corpus, relevant_docs


def evaluate_model(model, model_name, queries, corpus, relevant_docs):
    """Evaluate a model and return metrics"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"{'='*60}")
    
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=model_name.replace('/', '-'),
        show_progress_bar=True,
        batch_size=32
    )
    
    metrics = evaluator(model)
    return metrics


def main():
    print("="*70)
    print("MPNET BENCHMARK: Fine-tuned vs Base Model")
    print("="*70)
    
    EVAL_PATH = "data/mpnet_training/eval/eval_pairs.jsonl"
    FINETUNED_PATH = "models/enarm-mpnet-v2"
    BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"
    
    # Check paths
    if not Path(EVAL_PATH).exists():
        logger.error(f"Eval data not found: {EVAL_PATH}")
        return
    
    if not Path(FINETUNED_PATH).exists():
        logger.error(f"Fine-tuned model not found: {FINETUNED_PATH}")
        return
    
    # Load eval data
    queries, corpus, relevant_docs = load_eval_data(EVAL_PATH)
    
    # Load models
    logger.info("\nLoading models...")
    base_model = SentenceTransformer(BASE_MODEL)
    logger.info(f"Base model loaded: {BASE_MODEL}")
    
    finetuned_model = SentenceTransformer(FINETUNED_PATH)
    logger.info(f"Fine-tuned model loaded: {FINETUNED_PATH}")
    
    # Evaluate both
    base_metrics = evaluate_model(base_model, "base-mpnet", queries, corpus, relevant_docs)
    finetuned_metrics = evaluate_model(finetuned_model, "enarm-mpnet-v2", queries, corpus, relevant_docs)
    
    # Print comparison
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    key_metrics = [
        ('accuracy@1', 'Accuracy@1'),
        ('accuracy@5', 'Accuracy@5'),
        ('accuracy@10', 'Accuracy@10'),
        ('ndcg@10', 'NDCG@10'),
        ('mrr@10', 'MRR@10'),
        ('map@100', 'MAP@100'),
    ]
    
    results = {
        'base_model': BASE_MODEL,
        'finetuned_model': FINETUNED_PATH,
        'eval_samples': len(queries),
        'metrics': {}
    }
    
    print(f"\n{'Metric':<20} {'Base MPNet':<15} {'ENARM-MPNet-v2':<15} {'Improvement':<15}")
    print("-" * 65)
    
    for metric_key, metric_name in key_metrics:
        # Find the metric in results (with cosine prefix)
        base_val = None
        ft_val = None
        
        for k, v in base_metrics.items():
            if metric_key in k.lower():
                base_val = v
                break
        
        for k, v in finetuned_metrics.items():
            if metric_key in k.lower():
                ft_val = v
                break
        
        if base_val is not None and ft_val is not None:
            improvement = ft_val - base_val
            pct_improvement = (improvement / base_val * 100) if base_val > 0 else 0
            
            print(f"{metric_name:<20} {base_val:<15.4f} {ft_val:<15.4f} {improvement:+.4f} ({pct_improvement:+.1f}%)")
            
            results['metrics'][metric_name] = {
                'base': base_val,
                'finetuned': ft_val,
                'improvement': improvement,
                'improvement_pct': pct_improvement
            }
    
    # Save results
    output_file = Path("results/mpnet_benchmark_comparison.json")
    output_file.parent.mkdir(exist_ok=True)
    
    results['timestamp'] = str(datetime.now())
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVED] Results saved to: {output_file}")
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
