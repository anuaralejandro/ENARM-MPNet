#!/usr/bin/env python3
"""
Fine-tune MPNet for ENARM Medical Semantic Search
==================================================

Uses sentence-transformers with MultipleNegativesRankingLoss
for optimal semantic search performance.

Usage:
    conda activate enarmgpu
    python finetune_mpnet_semantic_search.py

Output:
    models/enarm-mpnet-v2/
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

# sentence-transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# datasets
from datasets import load_dataset, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mpnet_finetuning.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_training_data(train_path: str, eval_path: str):
    """Load training and evaluation datasets"""
    logger.info(f"Loading training data from {train_path}")
    
    train_dataset = load_dataset('json', data_files=train_path, split='train')
    eval_dataset = load_dataset('json', data_files=eval_path, split='train')
    
    logger.info(f"Train: {len(train_dataset):,} examples")
    logger.info(f"Eval: {len(eval_dataset):,} examples")
    
    return train_dataset, eval_dataset


def create_ir_evaluator(eval_dataset, model):
    """
    Create Information Retrieval evaluator for semantic search.
    
    This evaluates how well the model retrieves the correct document
    given a query (anchor -> positive matching).
    """
    # Build queries and corpus from eval set
    queries = {}
    corpus = {}
    relevant_docs = {}
    
    for i, example in enumerate(eval_dataset):
        query_id = f"q{i}"
        doc_id = f"d{i}"
        
        queries[query_id] = example['anchor']
        corpus[doc_id] = example['positive']
        relevant_docs[query_id] = {doc_id}  # Each query has 1 relevant doc
    
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="enarm-ir",
        show_progress_bar=True,
        batch_size=32
    )
    
    return evaluator


def main():
    print("="*70)
    print("MPNET FINE-TUNING FOR ENARM SEMANTIC SEARCH")
    print("="*70)
    
    # === Configuration ===
    BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"
    OUTPUT_DIR = "models/enarm-mpnet-v2"
    TRAIN_PATH = "data/mpnet_training/pairs_mnrl.jsonl"
    EVAL_PATH = "data/mpnet_training/eval/eval_pairs.jsonl"
    
    # Training hyperparameters
    EPOCHS = 2
    BATCH_SIZE = 32  # Adjust based on GPU memory (RTX 4070 = 12GB)
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    
    # === Check paths ===
    if not Path(TRAIN_PATH).exists():
        logger.error(f"Training data not found: {TRAIN_PATH}")
        logger.info("Run: python generate_training_dataset.py first")
        return
    
    # === Load model ===
    logger.info(f"\nLoading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)
    logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # === Load data ===
    train_dataset, eval_dataset = load_training_data(TRAIN_PATH, EVAL_PATH)
    
    # === Setup loss function ===
    # MultipleNegativesRankingLoss is ideal for semantic search
    # It uses in-batch negatives: for each (anchor, positive) pair,
    # all other positives in the batch serve as negatives
    loss = MultipleNegativesRankingLoss(model)
    logger.info("Loss function: MultipleNegativesRankingLoss")
    
    # === Setup evaluator ===
    evaluator = create_ir_evaluator(eval_dataset, model)
    
    # === Training arguments ===
    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        fp16=True,  # Enable FP16 for faster training on GPU
        bf16=False,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_enarm-ir_cosine_ndcg@10",
        logging_steps=100,
        logging_first_step=True,
        report_to=[],  # Disable wandb/tensorboard
    )
    
    logger.info(f"\nTraining configuration:")
    logger.info(f"  Epochs: {EPOCHS}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    logger.info(f"  Output: {OUTPUT_DIR}")
    
    # === Create trainer ===
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    
    # === Train ===
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)
    
    trainer.train()
    
    # === Save final model ===
    logger.info(f"\nSaving model to {OUTPUT_DIR}")
    model.save(OUTPUT_DIR)
    
    # === Final evaluation ===
    logger.info("\nRunning final evaluation...")
    final_metrics = evaluator(model)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"\nFinal metrics:")
    for k, v in final_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # Save training info
    info = {
        'base_model': BASE_MODEL,
        'output_dir': OUTPUT_DIR,
        'train_examples': len(train_dataset),
        'eval_examples': len(eval_dataset),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'final_metrics': final_metrics,
        'timestamp': str(datetime.now())
    }
    
    with open(Path(OUTPUT_DIR) / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n[DONE] Model saved to: {OUTPUT_DIR}")
    print("Ready for use in RAG service!")


if __name__ == "__main__":
    main()
