#!/usr/bin/env python3
"""
🧪 Evaluation Script for Fine-Tuned MPNet ENARM
================================================

Compares base all-mpnet-base-v2 vs fine-tuned enarm-mpnet-v1

Usage:
    conda activate enarmgpu
    python evaluate_mpnet_enarm.py
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ==================== TEST CASES ====================

# Format: (query, expected_match, expected_non_match)
# The model should score query↔expected_match HIGHER than query↔expected_non_match

TEST_CASES = [
    # Diabetes
    ("tratamiento farmacológico de diabetes tipo 2",
     "manejo de la DM2 con metformina",
     "tratamiento de la artritis reumatoide"),
    
    # Cardiología
    ("síntomas de infarto agudo al miocardio",
     "cuadro clínico del IAM con elevación del ST",
     "diagnóstico de gastritis erosiva"),
    
    # Neumología
    ("tratamiento de crisis asmática en pediatría",
     "manejo de exacerbación de asma en niños",
     "cálculos renales en adulto mayor"),
    
    # Neurología
    ("tratamiento del dolor neuropático diabético",
     "manejo de neuropatía diabética con gabapentina",
     "vacunación en el embarazo"),
    
    # Infectología
    ("antibiótico de primera línea en neumonía adquirida en comunidad",
     "tratamiento NAC con amoxicilina/ácido clavulánico",
     "manejo del hipotiroidismo"),
    
    # Pediatría
    ("vacunación en el recién nacido",
     "esquema de vacunación BCG y hepatitis B al nacimiento",
     "tratamiento de la hipertensión arterial"),
    
    # Ginecología
    ("preeclampsia severa manejo",
     "tratamiento de preeclampsia con sulfato de magnesio",
     "diagnóstico de EPOC"),
    
    # Urgencias
    ("reanimación cardiopulmonar en adulto",
     "RCP básica y avanzada protocolo ACLS",
     "alimentación complementaria en lactante"),
    
    # Abreviaturas médicas mexicanas
    ("contraindicaciones de los AINEs",
     "efectos adversos antiinflamatorios no esteroideos",
     "fisiopatología de la ictericia neonatal"),
    
    # Términos ENARM específicos
    ("primera línea para H. pylori",
     "triple terapia IBP + claritromicina + amoxicilina",
     "tratamiento de la psoriasis")
]

# ==================== EVALUATION ====================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def evaluate_model(model: SentenceTransformer, model_name: str) -> dict:
    """Evaluate model on test cases"""
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Evaluating: {model_name}")
    logger.info(f"{'='*60}")
    
    correct = 0
    total = len(TEST_CASES)
    similarities_match = []
    similarities_non_match = []
    
    for query, expected_match, expected_non_match in TEST_CASES:
        # Get embeddings
        emb_query = model.encode(query)
        emb_match = model.encode(expected_match)
        emb_non_match = model.encode(expected_non_match)
        
        # Calculate similarities
        sim_match = cosine_similarity(emb_query, emb_match)
        sim_non_match = cosine_similarity(emb_query, emb_non_match)
        
        similarities_match.append(sim_match)
        similarities_non_match.append(sim_non_match)
        
        # Check if model correctly ranks match > non_match
        is_correct = sim_match > sim_non_match
        if is_correct:
            correct += 1
        
        # Log results
        status = "✅" if is_correct else "❌"
        logger.info(f"\n{status} Query: '{query[:50]}...'")
        logger.info(f"   Match:     {sim_match:.4f} | '{expected_match[:40]}...'")
        logger.info(f"   Non-match: {sim_non_match:.4f} | '{expected_non_match[:40]}...'")
        logger.info(f"   Margin: {sim_match - sim_non_match:+.4f}")
    
    # Summary statistics
    accuracy = correct / total
    avg_match = np.mean(similarities_match)
    avg_non_match = np.mean(similarities_non_match)
    avg_margin = avg_match - avg_non_match
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📈 SUMMARY: {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"   Accuracy (correct ranking): {correct}/{total} = {accuracy:.1%}")
    logger.info(f"   Avg similarity (match):     {avg_match:.4f}")
    logger.info(f"   Avg similarity (non-match): {avg_non_match:.4f}")
    logger.info(f"   Avg margin:                 {avg_margin:+.4f}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'avg_match': avg_match,
        'avg_non_match': avg_non_match,
        'avg_margin': avg_margin,
        'correct': correct,
        'total': total
    }


def main():
    print("\n" + "="*60)
    print("🧪 MPNet Evaluation: Base vs Fine-Tuned")
    print("="*60)
    
    # Load models
    logger.info("\n📥 Loading models...")
    
    base_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    logger.info("   ✅ Base model loaded")
    
    try:
        finetuned_model = SentenceTransformer('models/enarm-mpnet-v1')
        logger.info("   ✅ Fine-tuned model loaded")
        has_finetuned = True
    except Exception as e:
        logger.warning(f"   ⚠️ Could not load fine-tuned model: {e}")
        has_finetuned = False
    
    # Evaluate base model
    base_results = evaluate_model(base_model, "Base MPNet (all-mpnet-base-v2)")
    
    # Evaluate fine-tuned model
    if has_finetuned:
        finetuned_results = evaluate_model(finetuned_model, "Fine-Tuned (enarm-mpnet-v1)")
        
        # Comparison
        print("\n" + "="*60)
        print("📊 COMPARISON: Base vs Fine-Tuned")
        print("="*60)
        
        acc_diff = finetuned_results['accuracy'] - base_results['accuracy']
        margin_diff = finetuned_results['avg_margin'] - base_results['avg_margin']
        
        print(f"\n{'Metric':<25} {'Base':<12} {'Fine-Tuned':<12} {'Δ Change':<12}")
        print("-" * 60)
        print(f"{'Accuracy':<25} {base_results['accuracy']:.1%}{'':<5} {finetuned_results['accuracy']:.1%}{'':<5} {acc_diff:+.1%}")
        print(f"{'Avg Match Similarity':<25} {base_results['avg_match']:.4f}{'':<4} {finetuned_results['avg_match']:.4f}{'':<4} {finetuned_results['avg_match'] - base_results['avg_match']:+.4f}")
        print(f"{'Avg Non-Match Similarity':<25} {base_results['avg_non_match']:.4f}{'':<4} {finetuned_results['avg_non_match']:.4f}{'':<4} {finetuned_results['avg_non_match'] - base_results['avg_non_match']:+.4f}")
        print(f"{'Avg Margin':<25} {base_results['avg_margin']:.4f}{'':<4} {finetuned_results['avg_margin']:.4f}{'':<4} {margin_diff:+.4f}")
        
        # Verdict
        print("\n" + "="*60)
        if finetuned_results['accuracy'] > base_results['accuracy']:
            print("✅ VERDICT: Fine-tuned model is BETTER for ENARM queries!")
        elif finetuned_results['accuracy'] == base_results['accuracy']:
            if finetuned_results['avg_margin'] > base_results['avg_margin']:
                print("✅ VERDICT: Fine-tuned model has better margins (more confident)")
            else:
                print("⚠️ VERDICT: Models perform similarly")
        else:
            print("⚠️ VERDICT: Base model performed better on these test cases")
            print("   Consider more epochs or different training strategy")
        print("="*60)


if __name__ == "__main__":
    main()
