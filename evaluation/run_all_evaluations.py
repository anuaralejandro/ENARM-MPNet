#!/usr/bin/env python3
"""
Master Evaluation Runner (Fixed)
=================================

Runs all evaluation scripts in sequence and generates a summary report.
Uses the CLEANED dataset and includes ALL evaluation scripts.

Usage:
    conda activate enarmgpu
    python evaluation/run_all_evaluations.py
"""

import subprocess
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_script(script_path: str, description: str) -> bool:
    """Run a Python script and return success status"""
    logger.info(f"\n{'='*70}")
    logger.info(f"[RUN] {description}")
    logger.info(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            # No timeout - let it run as long as needed
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            logger.info(f"[OK] {description} completed successfully")
            return True
        else:
            logger.error(f"[FAIL] {description} failed")
            logger.error(f"Error output:\n{result.stderr[:500]}")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] {description} failed: {e}")
        return False
    except Exception as e:
        logger.error(f"[ERROR] {description} failed with exception: {e}")
        return False


def generate_summary_report(results_dir: Path):
    """Generate a summary report from all evaluation results"""
    logger.info("\n" + "="*70)
    logger.info("[SUMMARY] Generating Summary Report")
    logger.info("="*70)
    
    summary = {
        'evaluation_date': datetime.now().isoformat(),
        'metrics': {}
    }
    
    # Load available metrics
    metrics_files = [
        ('01_ranking_metrics.json', 'ranking'),
        ('02_retrieval_metrics.json', 'retrieval'),
        ('03_specialty_metrics.json', 'specialty'),
        ('mpnet_benchmark_comparison.json', 'benchmark')
    ]
    
    for filename, key in metrics_files:
        filepath = results_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    summary['metrics'][key] = json.load(f)
                logger.info(f"  [OK] Loaded {filename}")
            except Exception as e:
                logger.warning(f"  [WARN] Could not load {filename}: {e}")
    
    # Save summary
    with open(results_dir / 'SUMMARY.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[SAVED] Summary saved to: {results_dir / 'SUMMARY.json'}")


def main():
    print("\n" + "="*70)
    print("ENARM-MPNet Evaluation Suite - Master Runner")
    print("="*70)
    print("\nThis will run all evaluations in sequence.")
    print("="*70)
    
    # Define ALL evaluation scripts (including 03!)
    evaluations = [
        ("evaluation/01_ranking_evaluation.py", "01 - Ranking Accuracy Evaluation"),
        ("evaluation/02_rag_retrieval_evaluation.py", "02 - RAG Retrieval Performance"),
        ("evaluation/03_specialty_analysis.py", "03 - Specialty Analysis"),
        ("evaluation/04_embedding_visualization.py", "04 - Embedding Visualization")
    ]
    
    results = {}
    start_time = datetime.now()
    
    # Run each evaluation
    for script_path, description in evaluations:
        if not Path(script_path).exists():
            logger.error(f"[MISSING] Script not found: {script_path}")
            results[description] = "[MISSING]"
            continue
            
        success = run_script(script_path, description)
        results[description] = "[OK]" if success else "[FAIL]"
    
    # Print final status
    print("\n" + "="*70)
    print("EVALUATION STATUS")
    print("="*70)
    for description, status in results.items():
        print(f"  {status} {description}")
    
    # Generate summary report
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    generate_summary_report(results_dir)
    
    elapsed = datetime.now() - start_time
    print(f"\nTotal time: {elapsed}")
    print("="*70)
    
    # Final message
    all_passed = all(status == "[OK]" for status in results.values())
    if all_passed:
        print("\n[SUCCESS] All evaluations completed!")
    else:
        print("\n[WARNING] Some evaluations failed. Check logs above.")
    
    print("="*70)


if __name__ == "__main__":
    main()
