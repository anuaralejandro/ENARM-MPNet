#!/usr/bin/env python3
"""
Training Dataset Generator for Sentence Transformers
=====================================================

Generates proper training data for fine-tuning MPNet for semantic search.

Task: Given a medical question query, find the most relevant flashcard.
      ("query", "positive_document") pairs for MultipleNegativesRankingLoss

Output formats:
1. JSONL for HuggingFace datasets
2. CSV for easy inspection
3. HuggingFace Dataset directly

Usage:
    python generate_training_dataset.py
"""

import json
import csv
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
from tqdm import tqdm

# For similarity-based negative mining
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("[WARNING] numpy/sklearn/sentence_transformers not found. Using random negatives.")


def load_cleaned_dataset(path: str) -> List[Dict]:
    """Load cleaned flashcards dataset"""
    print(f"\n[LOAD] Loading cleaned dataset from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flashcards = data.get('flashcards', [])
    print(f"[OK] Loaded {len(flashcards):,} flashcards")
    return flashcards


def create_training_pairs(flashcards: List[Dict]) -> List[Dict]:
    """
    Create (query, positive) pairs for MultipleNegativesRankingLoss.
    
    For semantic search training:
    - anchor = query (question reformulation or paraphrase)
    - positive = document (question + answer text)
    
    Strategy:
    1. Use question as anchor, Q+A as positive (retrieval perspective)
    2. Use question variations/paraphrases where natural
    """
    print("\n[GEN] Generating training pairs...")
    
    training_data = []
    
    for fc in tqdm(flashcards, desc="Creating pairs"):
        pregunta = fc.get('pregunta', '').strip()
        respuesta = fc.get('respuesta', '').strip()
        especialidad = fc.get('especialidad', '')
        
        if not pregunta or not respuesta:
            continue
        
        # Document = combined Q+A (what we embed in the database)
        document = f"{pregunta} {respuesta}"[:512]  # Max 512 chars
        
        # === PAIR TYPE 1: Question -> Document ===
        # This teaches the model that a question should retrieve its Q+A pair
        training_data.append({
            'anchor': pregunta,
            'positive': document,
            'especialidad': especialidad,
            'pair_type': 'question_to_qa'
        })
        
        # === PAIR TYPE 2: Answer fragment -> Document (harder examples) ===
        # This teaches that answer-related queries should also retrieve the document
        # Only for longer answers
        if len(respuesta) > 100:
            # Take first sentence as "answer query"
            first_sentence = respuesta.split('.')[0].strip()
            if len(first_sentence) > 20:
                training_data.append({
                    'anchor': first_sentence,
                    'positive': document,
                    'especialidad': especialidad,
                    'pair_type': 'answer_fragment_to_qa'
                })
        
        # === PAIR TYPE 3: Question + specialty context ===
        # This teaches specialty-aware retrieval
        if especialidad:
            query_with_context = f"{especialidad}: {pregunta}"
            training_data.append({
                'anchor': query_with_context,
                'positive': document,
                'especialidad': especialidad,
                'pair_type': 'specialty_query_to_qa'
            })
    
    print(f"[OK] Generated {len(training_data):,} training pairs")
    return training_data


def create_triplets_with_hard_negatives(
    flashcards: List[Dict],
    model_name: str = 'sentence-transformers/all-mpnet-base-v2',
    negatives_per_positive: int = 1
) -> List[Dict]:
    """
    Create (anchor, positive, negative) triplets with hard negatives.
    
    Hard negatives = from same specialty but different topic
    This forces model to learn fine-grained medical distinctions.
    """
    print("\n[GEN] Generating triplets with hard negatives...")
    
    if not HAS_EMBEDDINGS:
        print("[SKIP] Embeddings not available, returning empty")
        return []
    
    # Group by specialty
    by_specialty = defaultdict(list)
    for i, fc in enumerate(flashcards):
        esp = fc.get('especialidad', 'General')
        by_specialty[esp].append((i, fc))
    
    triplets = []
    
    for esp, fc_list in tqdm(by_specialty.items(), desc="Processing specialties"):
        if len(fc_list) < 3:  # Need at least 3 for triplets
            continue
        
        for idx, fc in fc_list:
            pregunta = fc.get('pregunta', '').strip()
            respuesta = fc.get('respuesta', '').strip()
            
            if not pregunta or not respuesta:
                continue
            
            document = f"{pregunta} {respuesta}"[:512]
            
            # Get negatives from SAME specialty (hard negatives)
            other_fcs = [f for i, f in fc_list if i != idx]
            
            # Sample negatives
            num_neg = min(negatives_per_positive, len(other_fcs))
            negatives = random.sample(other_fcs, num_neg)
            
            for neg_fc in negatives:
                neg_pregunta = neg_fc.get('pregunta', '').strip()
                neg_respuesta = neg_fc.get('respuesta', '').strip()
                neg_document = f"{neg_pregunta} {neg_respuesta}"[:512]
                
                triplets.append({
                    'anchor': pregunta,
                    'positive': document,
                    'negative': neg_document,
                    'especialidad': esp
                })
    
    print(f"[OK] Generated {len(triplets):,} triplets with hard negatives")
    return triplets


def save_for_sentence_transformers(
    pairs: List[Dict],
    triplets: List[Dict],
    output_dir: str
):
    """
    Save datasets in formats ready for sentence-transformers training.
    
    Creates:
    1. pairs_mnrl.jsonl - For MultipleNegativesRankingLoss (anchor, positive)
    2. triplets_mnrl.jsonl - For TripletLoss (anchor, positive, negative)
    3. pairs_cosent.jsonl - For CoSENTLoss (sentence1, sentence2, score)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # === 1. Pairs for MultipleNegativesRankingLoss ===
    # Format: {"anchor": "...", "positive": "..."}
    pairs_file = output_path / "pairs_mnrl.jsonl"
    with open(pairs_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            record = {
                'anchor': pair['anchor'],
                'positive': pair['positive']
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"[SAVE] {pairs_file}: {len(pairs):,} pairs")
    
    # === 2. Triplets for TripletLoss ===
    # Format: {"anchor": "...", "positive": "...", "negative": "..."}
    if triplets:
        triplets_file = output_path / "triplets_mnrl.jsonl"
        with open(triplets_file, 'w', encoding='utf-8') as f:
            for trip in triplets:
                record = {
                    'anchor': trip['anchor'],
                    'positive': trip['positive'],
                    'negative': trip['negative']
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"[SAVE] {triplets_file}: {len(triplets):,} triplets")
    
    # === 3. CoSENT format (for similarity loss) ===
    # Format: {"sentence1": "...", "sentence2": "...", "label": 1.0}
    # Positive pairs get label=1.0
    cosent_file = output_path / "pairs_cosent.jsonl"
    with open(cosent_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            record = {
                'sentence1': pair['anchor'],
                'sentence2': pair['positive'],
                'label': 1.0  # Positive similarity
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"[SAVE] {cosent_file}: {len(pairs):,} pairs with scores")
    
    # === 4. CSV for easy inspection ===
    csv_file = output_path / "training_pairs.csv"
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['anchor', 'positive', 'especialidad', 'pair_type'])
        for pair in pairs:
            writer.writerow([
                pair['anchor'][:200],
                pair['positive'][:200],
                pair.get('especialidad', ''),
                pair.get('pair_type', '')
            ])
    print(f"[SAVE] {csv_file}: {len(pairs):,} pairs (truncated for readability)")
    
    # === 5. Summary stats ===
    stats = {
        'total_pairs': len(pairs),
        'total_triplets': len(triplets) if triplets else 0,
        'pair_types': {}
    }
    
    for pair in pairs:
        pt = pair.get('pair_type', 'unknown')
        stats['pair_types'][pt] = stats['pair_types'].get(pt, 0) + 1
    
    stats_file = output_path / "dataset_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {stats_file}")
    
    print(f"\n[DONE] Training data saved to: {output_path}")


def split_train_eval(pairs: List[Dict], eval_ratio: float = 0.1) -> Tuple[List, List]:
    """Split data into train and evaluation sets"""
    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - eval_ratio))
    return pairs[:split_idx], pairs[split_idx:]


def main():
    print("="*70)
    print("SENTENCE TRANSFORMERS TRAINING DATASET GENERATOR")
    print("="*70)
    
    # Set seed for reproducibility
    random.seed(42)
    
    # Load cleaned dataset
    flashcards = load_cleaned_dataset('data/enarm_flashcards_cleaned_mpnet.json')
    
    # Create training pairs
    pairs = create_training_pairs(flashcards)
    
    # Create triplets with hard negatives (optional, for TripletLoss)
    # triplets = create_triplets_with_hard_negatives(flashcards)
    triplets = []  # Skip triplets for now, pairs are sufficient for MNRL
    
    # Split into train/eval
    train_pairs, eval_pairs = split_train_eval(pairs)
    
    print(f"\n[SPLIT] Train: {len(train_pairs):,} | Eval: {len(eval_pairs):,}")
    
    # Save training data
    save_for_sentence_transformers(
        pairs=train_pairs,
        triplets=triplets,
        output_dir='data/mpnet_training'
    )
    
    # Save eval data separately
    eval_output = Path('data/mpnet_training/eval')
    eval_output.mkdir(parents=True, exist_ok=True)
    
    eval_file = eval_output / "eval_pairs.jsonl"
    with open(eval_file, 'w', encoding='utf-8') as f:
        for pair in eval_pairs:
            record = {'anchor': pair['anchor'], 'positive': pair['positive']}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"[SAVE] {eval_file}: {len(eval_pairs):,} eval pairs")
    
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE!")
    print("="*70)
    print(f"\nFiles created in data/mpnet_training/:")
    print("  - pairs_mnrl.jsonl       (for MultipleNegativesRankingLoss)")
    print("  - pairs_cosent.jsonl     (for CoSENTLoss)")
    print("  - triplets_mnrl.jsonl    (for TripletLoss, if generated)")
    print("  - training_pairs.csv     (for manual inspection)")
    print("  - dataset_stats.json     (summary statistics)")
    print("  - eval/eval_pairs.jsonl  (evaluation set)")
    print("\nRecommended training:")
    print("  Loss: MultipleNegativesRankingLoss (semantic search)")
    print("  Base: sentence-transformers/all-mpnet-base-v2")
    print("  Epochs: 1-3")
    print("  Batch: 32 (or 16 if GPU memory limited)")


if __name__ == "__main__":
    main()
