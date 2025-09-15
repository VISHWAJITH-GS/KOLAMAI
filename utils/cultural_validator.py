"""
Cultural Pattern Validator
Ensures cultural authenticity and appropriateness in generated patterns.
Contains traditional pattern templates and cultural constraints.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json
import math


class CulturalTradition(Enum):
    """Supported cultural traditions"""
    ISLAMIC = "islamic"
    CELTIC = "celtic"
    MANDALA = "mandala"
    NATIVE_AMERICAN = "native_american"
    AFRICAN = "african"
    CHINESE = "chinese"
    JAPANESE = "japanese"
    INDIAN = "indian"
    AZTEC = "aztec"
    NORSE = "norse"
    ABORIGINAL = "aboriginal"
    POLYNESIAN = "polynesian"


@dataclass
class ValidationResult:
    """Result of cultural validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    confidence_score: float


@dataclass
class CulturalConstraint:
    """Represents a cultural constraint or rule"""
    name: str
    description: str
    constraint_type: str  # 'required', 'forbidden', 'preferred'
    parameters: Dict[str, Any]


@dataclass
class TraditionalElement:
    """Represents a traditional pattern element"""
    name: str
    tradition: CulturalTradition
    geometric_properties: Dict[str, Any]
    symbolic_meaning: str
    usage_rules: List[str]


class CulturalValidator:
    """Main class for validating cultural authenticity of patterns"""
    
    def __init__(self):
        self.traditional_templates = self._load_traditional_templates()
        self.cultural_constraints = self._load_cultural_constraints()
        self.sacred_symbols = self._load_sacred_symbols()
        self.color_meanings = self._load_color_meanings()
        
    def validate_traditional_rules(self, pattern_data: Dict[str, Any], 
                                 tradition: CulturalTradition) -> ValidationResult:
        """
        Checks adherence to traditional rules for a specific cultural tradition
        
        Args:
            pattern_data: The pattern data to validate
            tradition: The cultural tradition to validate against
            
        Returns:
            ValidationResult with details about rule compliance
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Get constraints for this tradition
        constraints = self.cultural_constraints.get(tradition, [])
        
        # Validate geometric rules
        geometric_validation = self._validate_geometric_rules(pattern_data, tradition)
        errors.extend(geometric_validation['errors'])
        warnings.extend(geometric_validation['warnings'])
        
        # Validate symmetry requirements
        symmetry_validation = self._validate_symmetry_rules(pattern_data, tradition)
        errors.extend(symmetry_validation['errors'])
        warnings.extend(symmetry_validation['warnings'])
        
        # Validate color usage
        color_validation = self._validate_color_rules(pattern_data, tradition)
        errors.extend(color_validation['errors'])
        warnings.extend(color_validation['warnings'])
        
        # Validate element combinations
        combination_validation = self._validate_element_combinations(pattern_data, tradition)
        errors.extend(combination_validation['errors'])
        warnings.extend(combination_validation['warnings'])
        
        # Validate proportions
        proportion_validation = self._validate_proportions(pattern_data, tradition)
        errors.extend(proportion_validation['errors'])
        warnings.extend(proportion_validation['warnings'])
        
        # Generate suggestions for improvement
        suggestions = self._generate_improvement_suggestions(pattern_data, tradition)
        
        # Calculate confidence score
        total_checks = len(constraints) + 5  # Base checks
        failed_checks = len(errors)
        confidence_score = max(0, (total_checks - failed_checks) / total_checks)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            confidence_score=confidence_score
        )
    
    def verify_cultural_elements(self, pattern_data: Dict[str, Any]) -> ValidationResult:
        """
        Validates cultural appropriateness and authenticity of pattern elements
        
        Args:
            pattern_data: The pattern data to validate
            
        Returns:
            ValidationResult with cultural appropriateness assessment
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check for sacred symbols misuse
        sacred_check = self._check_sacred_symbols(pattern_data)
        errors.extend(sacred_check['errors'])
        warnings.extend(sacred_check['warnings'])
        
        # Check for cultural mixing appropriateness
        mixing_check = self._check_cultural_mixing(pattern_data)
        warnings.extend(mixing_check['warnings'])
        suggestions.extend(mixing_check['suggestions'])
        
        # Check for stereotypical representations
        stereotype_check = self._check_stereotypes(pattern_data)
        warnings.extend(stereotype_check['warnings'])
        suggestions.extend(stereotype_check['suggestions'])
        
        # Check for proper attribution context
        attribution_check = self._check_attribution(pattern_data)
        suggestions.extend(attribution_check['suggestions'])
        
        confidence_score = 1.0 - (len(errors) * 0.3 + len(warnings) * 0.1)
        confidence_score = max(0, min(1, confidence_score))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            confidence_score=confidence_score
        )
    
    def get_traditional_template(self, tradition: CulturalTradition, 
                               pattern_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a traditional pattern template
        
        Args:
            tradition: The cultural tradition
            pattern_type: Type of pattern (e.g., 'geometric', 'floral', 'animal')
            
        Returns:
            Template data or None if not found
        """
        templates = self.traditional_templates.get(tradition, {})
        return templates.get(pattern_type)
    
    def suggest_authentic_modifications(self, pattern_data: Dict[str, Any],
                                      tradition: CulturalTradition) -> List[str]:
        """
        Suggests modifications to make pattern more culturally authentic
        
        Args:
            pattern_data: Current pattern data
            tradition: Target cultural tradition
            
        Returns:
            List of suggested modifications
        """
        suggestions = []
        
        # Analyze current pattern against traditional rules
        validation = self.validate_traditional_rules(pattern_data, tradition)
        
        # Add specific suggestions based on tradition
        if tradition == CulturalTradition.ISLAMIC:
            suggestions.extend(self._suggest_islamic_modifications(pattern_data))
        elif tradition == CulturalTradition.MANDALA:
            suggestions.extend(self._suggest_mandala_modifications(pattern_data))
        elif tradition == CulturalTradition.CELTIC:
            suggestions.extend(self._suggest_celtic_modifications(pattern_data))
        elif tradition == CulturalTradition.NATIVE_AMERICAN:
            suggestions.extend(self._suggest_native_american_modifications(pattern_data))
        
        return suggestions
    
    def _load_traditional_templates(self) -> Dict[CulturalTradition, Dict[str, Any]]:
        """Load traditional pattern templates"""
        return {
            CulturalTradition.ISLAMIC: {
                'geometric': {
                    'base_polygons': [8, 12, 16],  # Common in Islamic patterns
                    'symmetry_types': ['rotational', 'reflection'],
                    'forbidden_elements': ['figurative', 'animal'],
                    'required_properties': {
                        'continuous': True,
                        'infinite_extension': True,
                        'mathematical_precision': True
                    }
                },
                'arabesque': {
                    'element_types': ['vine', 'leaf', 'geometric'],
                    'interlacing_required': True,
                    'growth_patterns': ['spiral', 'branching'],
                    'density_rules': {'min': 0.6, 'max': 0.9}
                }
            },
            CulturalTradition.MANDALA: {
                'traditional': {
                    'center_required': True,
                    'radial_symmetry': True,
                    'symmetry_orders': [4, 6, 8, 12],
                    'layers': {'min': 3, 'max': 12},
                    'element_types': ['lotus', 'deity', 'geometric', 'mantra'],
                    'color_significance': True
                }
            },
            CulturalTradition.CELTIC: {
                'knotwork': {
                    'continuous_line': True,
                    'no_beginning_end': True,
                    'over_under_pattern': True,
                    'crossing_rules': 'alternating',
                    'common_motifs': ['trinity', 'spiral', 'interlace']
                },
                'spiral': {
                    'types': ['single', 'double', 'triple'],
                    'direction_significance': True,
                    'growth_pattern': 'logarithmic',
                    'sacred_numbers': [3, 7, 9]
                }
            },
            CulturalTradition.NATIVE_AMERICAN: {
                'geometric': {
                    'nature_based': True,
                    'tribal_specific': True,
                    'sacred_directions': [4, 6],  # 4 cardinal, 6 including up/down
                    'color_meanings': True,
                    'seasonal_associations': True
                }
            }
        }
    
    def _load_cultural_constraints(self) -> Dict[CulturalTradition, List[CulturalConstraint]]:
        """Load cultural constraints and rules"""
        return {
            CulturalTradition.ISLAMIC: [
                CulturalConstraint(
                    "no_figurative",
                    "Islamic geometric patterns traditionally avoid figurative representations",
                    "forbidden",
                    {"element_types": ["human", "animal", "figurative"]}
                ),
                CulturalConstraint(
                    "mathematical_precision",
                    "Patterns should demonstrate mathematical precision and infinite extension",
                    "required",
                    {"precision": "high", "tileable": True}
                ),
                CulturalConstraint(