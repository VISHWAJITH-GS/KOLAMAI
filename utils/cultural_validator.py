"""
Cultural Authenticity Validation System for Kolam Patterns

This module validates Kolam patterns against traditional Tamil cultural rules,
geometric constraints, and regional variations to ensure authenticity.
"""

import numpy as np
import json
import math
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import cv2
from pathlib import Path

class PatternType(Enum):
    """Enum for different types of Kolam patterns"""
    PULLI_KOLAM = "pulli_kolam"  # Dot-based patterns
    SIKKU_KOLAM = "sikku_kolam"  # Line-based patterns
    KAMBI_KOLAM = "kambi_kolam"  # Wire-like patterns
    RANGOLI = "rangoli"          # Free-form patterns
    FESTIVAL_SPECIAL = "festival_special"  # Festival-specific patterns

class SeasonalContext(Enum):
    """Seasonal appropriateness for patterns"""
    DAILY = "daily"
    FESTIVAL = "festival"
    MARGAZHI = "margazhi"  # Tamil month (Dec-Jan)
    THAI = "thai"          # Tamil month (Jan-Feb)
    WEDDING = "wedding"
    HARVEST = "harvest"

@dataclass
class CulturalRule:
    """Represents a traditional Kolam rule"""
    rule_id: str
    description: str
    pattern_types: List[PatternType]
    weight: float
    required: bool = False

@dataclass
class ValidationResult:
    """Result of cultural validation"""
    is_authentic: bool
    authenticity_score: float
    violations: List[str]
    recommendations: List[str]
    cultural_significance: str
    regional_match: Optional[str]

class CulturalValidator:
    """Main class for validating cultural authenticity of Kolam patterns"""
    
    def __init__(self, cultural_db_path: str = None):
        """Initialize the cultural validator with rules and constraints"""
        self.cultural_db_path = cultural_db_path or "data/cultural_database.json"
        self.load_cultural_database()
        self.initialize_traditional_rules()
        self.load_regional_variations()
        
    def load_cultural_database(self):
        """Load cultural information from JSON database"""
        try:
            with open(self.cultural_db_path, 'r', encoding='utf-8') as f:
                self.cultural_data = json.load(f)
        except FileNotFoundError:
            # Default cultural data if file not found
            self.cultural_data = self._get_default_cultural_data()
    
    def initialize_traditional_rules(self):
        """Initialize traditional Kolam rules and constraints"""
        self.traditional_rules = {
            'symmetry_preservation': CulturalRule(
                'SYM_001', 
                'Kolam must maintain rotational or reflective symmetry',
                [PatternType.PULLI_KOLAM, PatternType.SIKKU_KOLAM],
                0.3, True
            ),
            'continuous_line': CulturalRule(
                'CONT_001',
                'Sikku kolam should form continuous unbroken lines',
                [PatternType.SIKKU_KOLAM],
                0.25, True
            ),
            'dot_grid_regularity': CulturalRule(
                'DOT_001',
                'Pulli kolam must follow regular dot grid pattern',
                [PatternType.PULLI_KOLAM],
                0.2, True
            ),
            'geometric_harmony': CulturalRule(
                'GEOM_001',
                'Pattern should maintain geometric harmony and balance',
                list(PatternType),
                0.15
            ),
            'cultural_symbols': CulturalRule(
                'SYMB_001',
                'Should incorporate traditional Tamil cultural symbols',
                list(PatternType),
                0.1
            )
        }
    
    def load_regional_variations(self):
        """Load regional pattern variations"""
        self.regional_patterns = {
            'tamil_nadu': {
                'characteristics': ['geometric precision', 'dot-based', 'symmetrical'],
                'preferred_ratios': [(1, 1), (3, 2), (4, 3)],
                'complexity_range': (3, 15)
            },
            'karnataka': {
                'characteristics': ['flowing lines', 'curved patterns', 'artistic'],
                'preferred_ratios': [(2, 1), (5, 3), (3, 1)],
                'complexity_range': (2, 20)
            },
            'andhra_pradesh': {
                'characteristics': ['elaborate designs', 'festival-oriented', 'colorful'],
                'preferred_ratios': [(1, 1), (2, 3), (1, 2)],
                'complexity_range': (5, 25)
            }
        }
    
    def validate_traditional_rules(self, pattern_data: Dict) -> Dict[str, Any]:
        """Validate pattern against traditional Kolam rules"""
        violations = []
        score = 1.0
        
        # Extract pattern features
        symmetry_score = self._check_symmetry(pattern_data)
        continuity_score = self._check_continuity(pattern_data)
        grid_score = self._check_grid_regularity(pattern_data)
        harmony_score = self._check_geometric_harmony(pattern_data)
        symbol_score = self._check_cultural_symbols(pattern_data)
        
        # Apply rule-based scoring
        if symmetry_score < 0.7:
            violations.append("Poor symmetry preservation")
            score -= 0.3
        
        if pattern_data.get('type') == PatternType.SIKKU_KOLAM.value:
            if continuity_score < 0.8:
                violations.append("Lines are not continuous")
                score -= 0.25
        
        if pattern_data.get('type') == PatternType.PULLI_KOLAM.value:
            if grid_score < 0.6:
                violations.append("Irregular dot grid pattern")
                score -= 0.2
        
        if harmony_score < 0.5:
            violations.append("Poor geometric harmony")
            score -= 0.15
        
        if symbol_score < 0.3:
            violations.append("Lacks traditional cultural elements")
            score -= 0.1
        
        return {
            'score': max(0.0, score),
            'violations': violations,
            'detailed_scores': {
                'symmetry': symmetry_score,
                'continuity': continuity_score,
                'grid_regularity': grid_score,
                'harmony': harmony_score,
                'symbols': symbol_score
            }
        }
    
    def check_symmetry_groups(self, pattern_data: Dict) -> Dict[str, Any]:
        """Check if pattern follows traditional symmetry groups"""
        symmetry_types = {
            'rotational': self._check_rotational_symmetry(pattern_data),
            'reflective': self._check_reflective_symmetry(pattern_data),
            'translational': self._check_translational_symmetry(pattern_data),
            'glide_reflection': self._check_glide_reflection(pattern_data)
        }
        
        # Traditional Kolam typically uses 2-fold, 4-fold, or 8-fold symmetry
        valid_rotations = [2, 4, 8]
        rotation_order = symmetry_types.get('rotational', {}).get('order', 1)
        
        is_valid = rotation_order in valid_rotations or symmetry_types['reflective']['present']
        
        return {
            'is_valid': is_valid,
            'symmetry_types': symmetry_types,
            'traditional_match': rotation_order in valid_rotations
        }
    
    def verify_cultural_elements(self, pattern_data: Dict) -> Dict[str, Any]:
        """Verify presence of traditional cultural elements"""
        cultural_elements = {
            'lotus_motif': self._detect_lotus_patterns(pattern_data),
            'geometric_shapes': self._detect_geometric_shapes(pattern_data),
            'connecting_lines': self._detect_connecting_lines(pattern_data),
            'dot_arrangements': self._detect_dot_arrangements(pattern_data),
            'spiral_elements': self._detect_spiral_elements(pattern_data)
        }
        
        authenticity_indicators = {
            'mathematical_precision': self._check_mathematical_precision(pattern_data),
            'traditional_proportions': self._check_traditional_proportions(pattern_data),
            'cultural_symbolism': self._analyze_cultural_symbolism(pattern_data)
        }
        
        return {
            'elements': cultural_elements,
            'indicators': authenticity_indicators,
            'cultural_score': self._calculate_cultural_score(cultural_elements, authenticity_indicators)
        }
    
    def validate_pattern(self, pattern_data: Dict, context: Dict = None) -> ValidationResult:
        """Main validation function that combines all checks"""
        context = context or {}
        
        # Perform all validation checks
        rule_validation = self.validate_traditional_rules(pattern_data)
        symmetry_validation = self.check_symmetry_groups(pattern_data)
        cultural_validation = self.verify_cultural_elements(pattern_data)
        
        # Calculate overall authenticity score
        weights = {'rules': 0.5, 'symmetry': 0.3, 'cultural': 0.2}
        
        overall_score = (
            rule_validation['score'] * weights['rules'] +
            (1.0 if symmetry_validation['is_valid'] else 0.5) * weights['symmetry'] +
            cultural_validation['cultural_score'] * weights['cultural']
        )
        
        # Determine authenticity threshold
        authenticity_threshold = 0.7
        is_authentic = overall_score >= authenticity_threshold
        
        # Collect all violations and recommendations
        violations = rule_validation['violations'].copy()
        recommendations = self._generate_recommendations(
            rule_validation, symmetry_validation, cultural_validation
        )
        
        # Determine cultural significance
        cultural_significance = self._determine_cultural_significance(
            pattern_data, cultural_validation
        )
        
        # Find regional match
        regional_match = self._find_regional_match(pattern_data)
        
        return ValidationResult(
            is_authentic=is_authentic,
            authenticity_score=overall_score,
            violations=violations,
            recommendations=recommendations,
            cultural_significance=cultural_significance,
            regional_match=regional_match
        )
    
    # Helper methods for specific checks
    def _check_symmetry(self, pattern_data: Dict) -> float:
        """Check overall symmetry score"""
        if 'contours' not in pattern_data:
            return 0.5
        
        contours = pattern_data['contours']
        if not contours:
            return 0.0
        
        # Calculate moments for symmetry analysis
        symmetry_scores = []
        for contour in contours:
            if len(contour) > 10:  # Valid contour
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    # Calculate normalized central moments
                    nu20 = moments['nu20']
                    nu02 = moments['nu02']
                    nu11 = moments['nu11']
                    
                    # Symmetry measure based on central moments
                    symmetry = 1.0 - abs(nu20 - nu02) / (nu20 + nu02 + 1e-7)
                    symmetry_scores.append(max(0, symmetry))
        
        return np.mean(symmetry_scores) if symmetry_scores else 0.0
    
    def _check_continuity(self, pattern_data: Dict) -> float:
        """Check line continuity for Sikku kolam"""
        if 'lines' not in pattern_data:
            return 0.5
        
        lines = pattern_data['lines']
        if not lines:
            return 0.0
        
        # Check for gaps and breaks in lines
        continuity_score = 1.0
        gap_threshold = 10  # pixels
        
        for i, line in enumerate(lines):
            if len(line) < 2:
                continue
            
            # Check internal continuity
            for j in range(len(line) - 1):
                distance = np.linalg.norm(
                    np.array(line[j]) - np.array(line[j + 1])
                )
                if distance > gap_threshold:
                    continuity_score -= 0.1
        
        return max(0.0, continuity_score)
    
    def _check_grid_regularity(self, pattern_data: Dict) -> float:
        """Check dot grid regularity for Pulli kolam"""
        if 'dots' not in pattern_data:
            return 0.5
        
        dots = pattern_data['dots']
        if len(dots) < 4:
            return 0.0
        
        # Calculate grid regularity
        x_coords = [dot[0] for dot in dots]
        y_coords = [dot[1] for dot in dots]
        
        # Check for regular spacing
        x_diffs = np.diff(sorted(set(x_coords)))
        y_diffs = np.diff(sorted(set(y_coords)))
        
        x_regularity = 1.0 - np.std(x_diffs) / (np.mean(x_diffs) + 1e-7)
        y_regularity = 1.0 - np.std(y_diffs) / (np.mean(y_diffs) + 1e-7)
        
        return (x_regularity + y_regularity) / 2.0
    
    def _check_geometric_harmony(self, pattern_data: Dict) -> float:
        """Check geometric harmony and balance"""
        if 'features' not in pattern_data:
            return 0.5
        
        features = pattern_data['features']
        
        # Golden ratio check
        golden_ratio = 1.618
        harmony_score = 0.0
        
        # Check aspect ratios
        if 'bbox' in features:
            width, height = features['bbox'][2:4]
            if height > 0:
                ratio = width / height
                golden_deviation = abs(ratio - golden_ratio) / golden_ratio
                harmony_score += max(0, 1.0 - golden_deviation)
        
        # Check symmetry balance
        if 'center_of_mass' in features and 'geometric_center' in features:
            com = np.array(features['center_of_mass'])
            gc = np.array(features['geometric_center'])
            balance = 1.0 - np.linalg.norm(com - gc) / 100.0  # normalized
            harmony_score += max(0, balance)
        
        return harmony_score / 2.0
    
    def _check_cultural_symbols(self, pattern_data: Dict) -> float:
        """Check for traditional cultural symbols"""
        symbol_score = 0.0
        
        # Check for traditional patterns (simplified)
        if 'pattern_type' in pattern_data:
            traditional_types = [t.value for t in PatternType]
            if pattern_data['pattern_type'] in traditional_types:
                symbol_score += 0.5
        
        # Check for geometric regularity (indicator of traditional design)
        if 'regularity_score' in pattern_data.get('features', {}):
            symbol_score += pattern_data['features']['regularity_score'] * 0.3
        
        # Check for complexity within traditional ranges
        if 'complexity' in pattern_data:
            complexity = pattern_data['complexity']
            if 3 <= complexity <= 20:  # Traditional range
                symbol_score += 0.2
        
        return min(1.0, symbol_score)
    
    # Additional helper methods (simplified implementations)
    def _check_rotational_symmetry(self, pattern_data: Dict) -> Dict:
        """Check rotational symmetry properties"""
        # Simplified implementation
        return {'present': True, 'order': 4}
    
    def _check_reflective_symmetry(self, pattern_data: Dict) -> Dict:
        """Check reflective symmetry properties"""
        return {'present': True, 'axes': 2}
    
    def _check_translational_symmetry(self, pattern_data: Dict) -> Dict:
        """Check translational symmetry"""
        return {'present': False}
    
    def _check_glide_reflection(self, pattern_data: Dict) -> Dict:
        """Check glide reflection symmetry"""
        return {'present': False}
    
    def _detect_lotus_patterns(self, pattern_data: Dict) -> bool:
        """Detect lotus motif patterns"""
        return 'lotus' in str(pattern_data).lower()
    
    def _detect_geometric_shapes(self, pattern_data: Dict) -> List[str]:
        """Detect geometric shapes in pattern"""
        return ['circle', 'square', 'triangle']  # Simplified
    
    def _detect_connecting_lines(self, pattern_data: Dict) -> bool:
        """Detect connecting lines between elements"""
        return 'lines' in pattern_data and len(pattern_data.get('lines', [])) > 0
    
    def _detect_dot_arrangements(self, pattern_data: Dict) -> Dict:
        """Detect dot arrangement patterns"""
        dots = pattern_data.get('dots', [])
        return {'count': len(dots), 'regular_grid': len(dots) > 4}
    
    def _detect_spiral_elements(self, pattern_data: Dict) -> bool:
        """Detect spiral elements"""
        return False  # Simplified
    
    def _check_mathematical_precision(self, pattern_data: Dict) -> float:
        """Check mathematical precision of the pattern"""
        return 0.8  # Simplified
    
    def _check_traditional_proportions(self, pattern_data: Dict) -> float:
        """Check if pattern follows traditional proportions"""
        return 0.7  # Simplified
    
    def _analyze_cultural_symbolism(self, pattern_data: Dict) -> float:
        """Analyze cultural symbolism in the pattern"""
        return 0.6  # Simplified
    
    def _calculate_cultural_score(self, elements: Dict, indicators: Dict) -> float:
        """Calculate overall cultural authenticity score"""
        element_score = sum(1 for v in elements.values() if v) / len(elements)
        indicator_score = sum(indicators.values()) / len(indicators)
        return (element_score + indicator_score) / 2.0
    
    def _generate_recommendations(self, rule_val: Dict, sym_val: Dict, cult_val: Dict) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        if rule_val['score'] < 0.7:
            recommendations.append("Improve overall geometric regularity")
        
        if not sym_val['is_valid']:
            recommendations.append("Add proper symmetry (2-fold, 4-fold, or 8-fold)")
        
        if cult_val['cultural_score'] < 0.5:
            recommendations.append("Incorporate more traditional Tamil elements")
        
        return recommendations
    
    def _determine_cultural_significance(self, pattern_data: Dict, cultural_val: Dict) -> str:
        """Determine cultural significance of the pattern"""
        pattern_type = pattern_data.get('pattern_type', 'unknown')
        
        significance_map = {
            'pulli_kolam': 'Traditional dot-based Kolam representing prosperity and protection',
            'sikku_kolam': 'Line-based Kolam symbolizing life\'s continuity and interconnectedness',
            'rangoli': 'Artistic pattern for festivals and celebrations',
            'festival_special': 'Special ceremonial pattern for religious occasions'
        }
        
        return significance_map.get(pattern_type, 'Contemporary artistic interpretation of Kolam tradition')
    
    def _find_regional_match(self, pattern_data: Dict) -> Optional[str]:
        """Find the best regional match for the pattern"""
        # Simplified regional matching
        complexity = pattern_data.get('complexity', 10)
        
        for region, props in self.regional_patterns.items():
            min_comp, max_comp = props['complexity_range']
            if min_comp <= complexity <= max_comp:
                return region
        
        return None
    
    def _get_default_cultural_data(self) -> Dict:
        """Return default cultural data if file not available"""
        return {
            "patterns": {
                "pulli_kolam": {
                    "description": "Dot-based traditional Kolam",
                    "significance": "Prosperity and protection",
                    "region": "Tamil Nadu"
                }
            },
            "symbols": {
                "lotus": "Purity and divine beauty",
                "swastika": "Prosperity and good fortune"
            }
        }