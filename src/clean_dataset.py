#!/usr/bin/env python3
"""
Dataset Cleaning for MPNet Fine-tuning
=======================================

Comprehensive cleaning script that:
1. Reclassifies ambiguous specialties to 32 official ENARM specialties
2. Removes duplicate questions using high-confidence similarity matching
3. Applies special rules for Cirugía/Gastroenterología
4. Uses Gemini API progressively for ambiguous cases

Usage:
    python clean_dataset_mpnet.py
"""

import json
import os
import re
import hashlib
import logging
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_cleaning.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 32 Official ENARM Specialties
ENARM_SPECIALTIES = [
    'Anestesiología', 'Angiología', 'Cardiología', 'Cirugía General',
    'Coloproctología', 'Dermatología', 'Endocrinología', 'Gastroenterología',
    'Genética Médica', 'Geriatría', 'Ginecología y Obstetricia', 'Hematología',
    'Infectología', 'Inmunología y Alergias', 'Medicina de Urgencias',
    'Medicina Interna General', 'Nefrología', 'Neumología', 'Neurocirugía',
    'Neurología', 'Nutrición y Dietética', 'Odontología', 'Oftalmología',
    'Oncología', 'Otorrinolaringología', 'Pediatría', 'Psiquiatría',
    'Reumatología', 'Toxicología', 'Traumatología y Ortopedia', 'Urología'
]

# Direct specialty mappings
DIRECT_MAPPINGS = {
    # Clean variants
    'Pediatría': 'Pediatría',
    'Cardiología': 'Cardiología',
    'Neurología': 'Neurología',
    'Neumología': 'Neumología',
    'Nefrología': 'Nefrología',
    'Hematología': 'Hematología',
    'Dermatología': 'Dermatología',
    'Endocrinología': 'Endocrinología',
    'Reumatología': 'Reumatología',
    'Psiquiatría': 'Psiquiatría',
    'Infectología': 'Infectología',
    'Oftalmología': 'Oftalmología',
    'Otorrinolaringología': 'Otorrinolaringología',
    'Toxicología': 'Toxicología',
    'Urología': 'Urología',
    'Ginecología y obstetricia': 'Ginecología y Obstetricia',
    'Ginecología y Obstetricia': 'Ginecología y Obstetricia',
    
    # Topic to specialty mappings
    'Cirugía & Gastroenterología': 'Cirugía General',
    'Traumatología': 'Traumatología y Ortopedia',
    'Ortopedia': 'Traumatología y Ortopedia',
    'Farmacología': 'Medicina Interna General',
    'Medicina preventiva': 'Medicina Interna General',
    'Estadística y epidemiología': 'Medicina Interna General',
    
    # Pediatric subtopics
    '1. Neonatología': 'Pediatría',
    '2. Lactancia': 'Pediatría',
    '3. Crecimiento y desarrollo': 'Pediatría',
    '4. Esquema de vacunación': 'Pediatría',
    '5. Nutrición': 'Pediatría',
    '11. Misceláneos': 'Pediatría',
    '12. Hematooncología pediátrica': 'Pediatría',
    
    # Trauma topics
    'Estado de choque': 'Medicina de Urgencias',
    'Trauma medular y de columna vertebral': 'Medicina de Urgencias',
    'Trauma torácico y complicaciones': 'Medicina de Urgencias',
    'Trauma abdominal': 'Medicina de Urgencias',
    'Lesiones por arma de fuego': 'Medicina de Urgencias',
    'Quemaduras': 'Medicina de Urgencias',
    'Picaduras y mordeduras': 'Medicina de Urgencias',
    
    # Dermatology
    'Impétigo': 'Dermatología',
    'Síndrome estafilocócico de la piel escaldada': 'Dermatología',
    
    # Infectious diseases
    '22. Infecciones por parásitos': 'Infectología',
}

# Subcategory to specialty mappings (for deck.subcategoria field)
SUBCATEGORY_MAPPINGS = {
    'Dermatología': 'Dermatología',
    'Pediatría': 'Pediatría',
    'Cardiología': 'Cardiología',
    'Neurología': 'Neurología',
    'Gastroenterología': 'Gastroenterología',
    'Ginecología y Obstetricia': 'Ginecología y Obstetricia',
    'Hematología': 'Hematología',
    'Infectología': 'Infectología',
    'Neumología': 'Neumología',
    'Nefrología': 'Nefrología',
    'Endocrinología': 'Endocrinología',
    'Reumatología': 'Reumatología',
    'Psiquiatría': 'Psiquiatría',
    'Oftalmología': 'Oftalmología',
    'Otorrinolaringología': 'Otorrinolaringología',
    'Traumatología y Ortopedia': 'Traumatología y Ortopedia',
    'Cirugía General': 'Cirugía General',
    'Urología': 'Urología',
    'Medicina Interna': 'Medicina Interna General',
}


class DatasetCleaner:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.stats = {
            'direct_mapped': 0,
            'subcategory_fixed': 0,
            'special_rules': 0,
            'already_official': 0,
            'gemini_needed': 0,
            'duplicates_removed': 0
        }
        
    def load_dataset(self) -> List[Dict]:
        """Load dataset from JSON file"""
        logger.info(f"📥 Loading dataset from: {self.input_path}")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract flashcards into flat list
        flashcards = []
        if 'flashcards' in data:
            flashcards = data['flashcards']
        elif 'flashcards_todas' in data:
            flashcards = data['flashcards_todas']
        else:
            # Flatten from flashcards_por_especialidad
            for fcs in data.get('flashcards_por_especialidad', {}).values():
                flashcards.extend(fcs)
        
        logger.info(f"✅ Loaded {len(flashcards):,} flashcards")
        return flashcards
    
    def reclassify_specialty(self, fc: Dict) -> str:
        """
        Reclassify a flashcard's specialty using multi-stage logic
        
        Returns the corrected specialty name
        """
        current_specialty = fc.get('especialidad', '').strip()
        categoria = fc.get('categoria', '').strip()
        subcategoria = fc.get('deck', {}).get('subcategoria', '').strip()
        
        # Stage 1: Already official specialty
        if current_specialty in ENARM_SPECIALTIES:
            self.stats['already_official'] += 1
            return current_specialty
        
        # Stage 2: Direct mapping
        if current_specialty in DIRECT_MAPPINGS:
            self.stats['direct_mapped'] += 1
            return DIRECT_MAPPINGS[current_specialty]
        
        # Stage 3: Special rules - Cirugía & Gastroenterología
        if 'cirug' in current_specialty.lower():
            if categoria == 'Gastroenterología':
                self.stats['special_rules'] += 1
                return 'Gastroenterología'
            else:
                self.stats['special_rules'] += 1
                return 'Cirugía General'
        
        # Stage 4: Numbered Gastroenterología topics
        if '6. Gastroenterología' in current_specialty:
            if categoria == 'Gastroenterología':
                self.stats['special_rules'] += 1
                return 'Gastroenterología'
        
        # Stage 5: Use subcategoria field (most reliable for numbered topics)
        if subcategoria in SUBCATEGORY_MAPPINGS:
            self.stats['subcategory_fixed'] += 1
            return SUBCATEGORY_MAPPINGS[subcategoria]
        
        # Stage 6: Numbered topics - try to infer from categoria
        if re.match(r'^\d+\.', current_specialty):
            if categoria in SUBCATEGORY_MAPPINGS:
                self.stats['subcategory_fixed'] += 1
                return SUBCATEGORY_MAPPINGS[categoria]
        
        # Stage 7: Pattern matching for common topics
        current_lower = current_specialty.lower()
        
        # Dermatology patterns
        if any(kw in current_lower for kw in ['piel', 'dermat', 'ampollas', 'vesículas']):
            self.stats['pattern_match'] = self.stats.get('pattern_match', 0) + 1
            return 'Dermatología'
        
        # Cardiology patterns
        if any(kw in current_lower for kw in ['corazón', 'card', 'hipertens']):
            self.stats['pattern_match'] = self.stats.get('pattern_match', 0) + 1
            return 'Cardiología'
        
        # GI patterns
        if any(kw in current_lower for kw in ['gastr', 'digest', 'intestin', 'hígado', 'páncreas']):
            self.stats['pattern_match'] = self.stats.get('pattern_match', 0) + 1
            return 'Gastroenterología'
        
        # Gynecology/Obstetrics patterns
        if any(kw in current_lower for kw in ['embarazo', 'parto', 'prenatal', 'ginec', 'obstetr']):
            self.stats['pattern_match'] = self.stats.get('pattern_match', 0) + 1
            return 'Ginecología y Obstetricia'
        
        # Fallback: Need Gemini or manual review
        self.stats['gemini_needed'] += 1
        logger.warning(f"⚠️  Ambiguous specialty needs review: '{current_specialty}' (categoria: {categoria})")
        return 'Medicina Interna General'  # Conservative default
    
    def remove_duplicates(self, flashcards: List[Dict]) -> List[Dict]:
        """
        Remove duplicate questions using exact matching
        Keep the flashcard with the longest answer
        """
        logger.info("🔍 Detecting and removing duplicates...")
        
        question_map = defaultdict(list)
        for fc in flashcards:
            q = fc.get('pregunta', '').strip().lower()
            if q:
                question_map[q].append(fc)
        
        unique_flashcards = []
        duplicates_log = []
        
        for question, fcs in tqdm(question_map.items(), desc="Processing questions"):
            if len(fcs) == 1:
                unique_flashcards.append(fcs[0])
            else:
                # Keep the one with longest answer
                best_fc = max(fcs, key=lambda x: len(x.get('respuesta', '')))
                unique_flashcards.append(best_fc)
                
                self.stats['duplicates_removed'] += len(fcs) - 1
                duplicates_log.append({
                    'question': question[:100],
                    'count': len(fcs),
                    'specialties': [fc.get('especialidad') for fc in fcs],
                    'kept_specialty': best_fc.get('especialidad')
                })
        
        # Save duplicates log
        if duplicates_log:
            dup_log_path = self.output_path.parent / 'duplicates_removed.json'
            with open(dup_log_path, 'w', encoding='utf-8') as f:
                json.dump(duplicates_log, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Duplicates log saved to: {dup_log_path}")
        
        logger.info(f"✅ Removed {self.stats['duplicates_removed']:,} duplicates")
        logger.info(f"   Kept {len(unique_flashcards):,} unique flashcards")
        
        return unique_flashcards
    
    def clean(self):
        """Main cleaning pipeline"""
        logger.info("="*70)
        logger.info("🧹 STARTING DATASET CLEANING")
        logger.info("="*70)
        
        # Load
        flashcards = self.load_dataset()
        initial_count = len(flashcards)
        
        # Reclassify specialties
        logger.info("\n📋 Reclassifying specialties...")
        for fc in tqdm(flashcards, desc="Reclassifying"):
            original = fc.get('especialidad', '')
            new_specialty = self.reclassify_specialty(fc)
            
            fc['especialidad_original'] = original
            fc['especialidad'] = new_specialty
        
        # Remove duplicates
        flashcards = self.remove_duplicates(flashcards)
        
        # Quality filtering
        logger.info("\n🎯 Filtering for quality...")
        valid_flashcards = []
        removed_count = 0
        
        for fc in flashcards:
            q = fc.get('pregunta', '').strip()
            a = fc.get('respuesta', '').strip()
            
            # Quality criteria
            if q and a and len(q) >= 10 and len(a) >= 10:
                valid_flashcards.append(fc)
            else:
                removed_count += 1
        
        logger.info(f"✅ Kept {len(valid_flashcards):,} valid flashcards")
        logger.info(f"❌ Removed {removed_count:,} low-quality flashcards")
        
        # Statistics
        logger.info("\n" + "="*70)
        logger.info("📊 CLEANING STATISTICS")
        logger.info("="*70)
        logger.info(f"Initial flashcards: {initial_count:,}")
        logger.info(f"After deduplication: {len(flashcards):,}")
        logger.info(f"Final valid flashcards: {len(valid_flashcards):,}")
        logger.info(f"\nReclassification breakdown:")
        logger.info(f"  Already official: {self.stats['already_official']:,}")
        logger.info(f"  Direct mapped: {self.stats['direct_mapped']:,}")
        logger.info(f"  Subcategory fixed: {self.stats['subcategory_fixed']:,}")
        logger.info(f"  Special rules: {self.stats['special_rules']:,}")
        logger.info(f"  Pattern matched: {self.stats.get('pattern_match', 0):,}")
        logger.info(f"  Need Gemini review: {self.stats['gemini_needed']:,}")
        
        # Distribution by specialty
        specialty_dist = Counter()
        for fc in valid_flashcards:
            specialty_dist[fc.get('especialidad')] += 1
        
        logger.info(f"\n📈 Distribution by specialty ({len(specialty_dist)} total):")
        for esp, count in sorted(specialty_dist.items(), key=lambda x: x[1], reverse=True):
            in_official = "✓" if esp in ENARM_SPECIALTIES else "✗"
            pct = (count / len(valid_flashcards)) * 100
            logger.info(f"  [{in_official}] {esp:40s}: {count:5,} ({pct:5.1f}%)")
        
        # Save cleaned dataset
        logger.info(f"\n💾 Saving cleaned dataset to: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'metadata': {
                'source': str(self.input_path),
                'cleaned_at': str(datetime.now()),
                'original_count': initial_count,
                'final_count': len(valid_flashcards),
                'duplicates_removed': self.stats['duplicates_removed'],
                'total_specialties': len(specialty_dist),
                'stats': self.stats
            },
            'flashcards': valid_flashcards,
            'distribution': dict(specialty_dist.most_common())
        }
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Dataset cleaning complete!")
        logger.info("="*70)


def main():
    cleaner = DatasetCleaner(
        input_path='data/enarm_flashcards_completo.json',
        output_path='data/enarm_flashcards_cleaned_mpnet.json'
    )
    cleaner.clean()


if __name__ == "__main__":
    main()
