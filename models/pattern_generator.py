import numpy as np
import json
import math
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Enum for different types of Kolam patterns"""
    PULLI_KOLAM = "pulli_kolam"  # Dot-based patterns
    SIKKU_KOLAM = "sikku_kolam"  # Line-based patterns
    KAMBI_KOLAM = "kambi_kolam"  # Wire/rope-like patterns
    CHUZHI_KOLAM = "chuzhi_kolam"  # Spiral patterns
    MANDALAM = "mandalam"  # Circular patterns

class SymmetryType(Enum):
    """Types of symmetry in Kolam patterns"""
    ROTATIONAL = "rotational"
    REFLECTIVE = "reflective" 
    TRANSLATIONAL = "translational"
    POINT = "point"

@dataclass
class Point:
    """Represents a point in 2D space"""
    x: float
    y: float
    
    def distance(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def rotate(self, angle: float, center: 'Point') -> 'Point':
        """Rotate point around center by given angle"""
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        dx, dy = self.x - center.x, self.y - center.y
        return Point(
            center.x + dx * cos_a - dy * sin_a,
            center.y + dx * sin_a + dy * cos_a
        )
    
    def reflect(self, axis_point: 'Point', axis_direction: Tuple[float, float]) -> 'Point':
        """Reflect point across a line"""
        # Normalize direction vector
        dx, dy = axis_direction
        length = math.sqrt(dx*dx + dy*dy)
        dx, dy = dx/length, dy/length
        
        # Vector from axis point to this point
        vx, vy = self.x - axis_point.x, self.y - axis_point.y
        
        # Project onto axis and reflect
        dot = vx * dx + vy * dy
        return Point(
            self.x - 2 * dot * dx,
            self.y - 2 * dot * dy
        )

@dataclass
class CulturalRules:
    """Defines traditional Kolam cultural rules and constraints"""
    # Traditional mathematical ratios (based on Tamil architectural principles)
    GOLDEN_RATIO: float = 1.618
    SACRED_RATIOS: List[float] = None
    
    # Grid constraints
    MIN_GRID_SIZE: int = 3
    MAX_GRID_SIZE: int = 25
    PREFERRED_GRID_SIZES: List[int] = None
    
    # Dot placement rules
    DOT_SPACING_RATIO: float = 1.0  # Equal spacing
    MIN_DOT_DISTANCE: float = 1.0
    
    # Cultural constraints
    AUSPICIOUS_NUMBERS: List[int] = None
    FORBIDDEN_NUMBERS: List[int] = None
    
    # Seasonal appropriateness
    FESTIVAL_PATTERNS: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.SACRED_RATIOS is None:
            self.SACRED_RATIOS = [1.0, 1.414, 1.618, 2.0, 2.414]  # 1, √2, φ, 2, 1+√2
        
        if self.PREFERRED_GRID_SIZES is None:
            self.PREFERRED_GRID_SIZES = [3, 5, 7, 9, 11, 13, 15]  # Odd numbers preferred
        
        if self.AUSPICIOUS_NUMBERS is None:
            self.AUSPICIOUS_NUMBERS = [3, 5, 7, 9, 11, 13, 15, 21]
        
        if self.FORBIDDEN_NUMBERS is None:
            self.FORBIDDEN_NUMBERS = [4, 8, 14]  # Generally avoided in Tamil culture
        
        if self.FESTIVAL_PATTERNS is None:
            self.FESTIVAL_PATTERNS = {
                "pongal": ["rice_pattern", "sugarcane_kolam", "sun_mandalam"],
                "diwali": ["deepam_kolam", "lotus_pattern", "prosperity_sikku"],
                "navaratri": ["goddess_kolam", "shakti_mandalam", "durga_pattern"],
                "thai": ["floral_kolam", "peacock_pattern", "geometric_sikku"]
            }

class SymmetryController:
    """Handles symmetry operations and constraints for Kolam patterns"""
    
    def __init__(self, symmetry_types: List[SymmetryType] = None):
        self.symmetry_types = symmetry_types or [SymmetryType.ROTATIONAL, SymmetryType.REFLECTIVE]
        self.symmetry_orders = {
            SymmetryType.ROTATIONAL: [2, 4, 6, 8],  # 2-fold, 4-fold, etc.
            SymmetryType.REFLECTIVE: [1, 2, 4],     # Number of reflection axes
            SymmetryType.POINT: [1],
            SymmetryType.TRANSLATIONAL: [1, 2]
        }
    
    def apply_rotational_symmetry(self, points: List[Point], center: Point, order: int) -> List[Point]:
        """Apply rotational symmetry of given order"""
        result = points.copy()
        angle_step = 2 * math.pi / order
        
        for i in range(1, order):
            angle = i * angle_step
            rotated_points = [p.rotate(angle, center) for p in points]
            result.extend(rotated_points)
        
        return result
    
    def apply_reflective_symmetry(self, points: List[Point], center: Point, num_axes: int) -> List[Point]:
        """Apply reflective symmetry with specified number of axes"""
        result = points.copy()
        
        for i in range(num_axes):
            angle = i * math.pi / num_axes
            axis_direction = (math.cos(angle), math.sin(angle))
            reflected_points = [p.reflect(center, axis_direction) for p in points]
            result.extend(reflected_points)
        
        return result
    
    def apply_point_symmetry(self, points: List[Point], center: Point) -> List[Point]:
        """Apply point symmetry (180-degree rotation)"""
        result = points.copy()
        reflected_points = [Point(2*center.x - p.x, 2*center.y - p.y) for p in points]
        result.extend(reflected_points)
        return result
    
    def validate_symmetry(self, points: List[Point], center: Point, tolerance: float = 0.1) -> Dict[SymmetryType, bool]:
        """Validate which symmetries are present in the point set"""
        validation_results = {}
        
        # Check rotational symmetry
        for order in [2, 4, 6, 8]:
            is_symmetric = self._check_rotational_symmetry(points, center, order, tolerance)
            validation_results[f"rotational_{order}"] = is_symmetric
        
        # Check reflective symmetry
        for num_axes in [1, 2, 4]:
            is_symmetric = self._check_reflective_symmetry(points, center, num_axes, tolerance)
            validation_results[f"reflective_{num_axes}"] = is_symmetric
        
        return validation_results
    
    def _check_rotational_symmetry(self, points: List[Point], center: Point, order: int, tolerance: float) -> bool:
        """Check if points have rotational symmetry of given order"""
        angle_step = 2 * math.pi / order
        
        for point in points:
            found_match = False
            for i in range(1, order):
                angle = i * angle_step
                rotated_point = point.rotate(angle, center)
                
                # Check if rotated point matches any existing point
                for existing_point in points:
                    if rotated_point.distance(existing_point) < tolerance:
                        found_match = True
                        break
                
                if not found_match:
                    return False
        
        return True
    
    def _check_reflective_symmetry(self, points: List[Point], center: Point, num_axes: int, tolerance: float) -> bool:
        """Check if points have reflective symmetry"""
        for i in range(num_axes):
            angle = i * math.pi / num_axes
            axis_direction = (math.cos(angle), math.sin(angle))
            
            for point in points:
                reflected_point = point.reflect(center, axis_direction)
                
                # Check if reflected point matches any existing point
                found_match = False
                for existing_point in points:
                    if reflected_point.distance(existing_point) < tolerance:
                        found_match = True
                        break
                
                if not found_match:
                    return False
        
        return True

class RuleEngine:
    """Applies traditional Kolam rules and validates cultural authenticity"""
    
    def __init__(self, cultural_rules: CulturalRules):
        self.rules = cultural_rules
        self.pattern_templates = self._load_pattern_templates()
    
    def _load_pattern_templates(self) -> Dict[str, Any]:
        """Load traditional pattern templates"""
        return {
            "basic_pulli": {
                "description": "Basic dot pattern with connecting lines",
                "min_dots": 9,
                "connectivity": "all_connected",
                "symmetry": "rotational"
            },
            "lotus_mandalam": {
                "description": "Lotus-inspired circular pattern",
                "petals": 8,
                "layers": 3,
                "symmetry": "rotational"
            },
            "peacock_kolam": {
                "description": "Stylized peacock pattern",
                "elements": ["body", "tail", "neck", "head"],
                "asymmetry_allowed": True
            }
        }
    
    def validate_grid_size(self, grid_size: int) -> Tuple[bool, str]:
        """Validate if grid size follows traditional rules"""
        if grid_size < self.rules.MIN_GRID_SIZE:
            return False, f"Grid size {grid_size} too small. Minimum: {self.rules.MIN_GRID_SIZE}"
        
        if grid_size > self.rules.MAX_GRID_SIZE:
            return False, f"Grid size {grid_size} too large. Maximum: {self.rules.MAX_GRID_SIZE}"
        
        if grid_size in self.rules.FORBIDDEN_NUMBERS:
            return False, f"Grid size {grid_size} is culturally inappropriate"
        
        if grid_size in self.rules.PREFERRED_GRID_SIZES:
            return True, f"Grid size {grid_size} is traditionally preferred"
        
        return True, f"Grid size {grid_size} is acceptable"
    
    def validate_dot_placement(self, dots: List[Point]) -> Tuple[bool, str]:
        """Validate dot placement according to traditional rules"""
        if len(dots) < 4:
            return False, "Too few dots for a meaningful pattern"
        
        # Check minimum distance between dots
        for i, dot1 in enumerate(dots):
            for j, dot2 in enumerate(dots[i+1:], i+1):
                distance = dot1.distance(dot2)
                if distance < self.rules.MIN_DOT_DISTANCE:
                    return False, f"Dots {i} and {j} too close (distance: {distance:.2f})"
        
        # Check if number of dots is auspicious
        num_dots = len(dots)
        if num_dots in self.rules.FORBIDDEN_NUMBERS:
            return False, f"{num_dots} dots is culturally inappropriate"
        
        return True, f"Dot placement with {num_dots} dots is valid"
    
    def apply_connectivity_rules(self, dots: List[Point], pattern_type: PatternType) -> List[Tuple[Point, Point]]:
        """Apply traditional connectivity rules between dots"""
        connections = []
        
        if pattern_type == PatternType.PULLI_KOLAM:
            # Traditional pulli kolam: connect dots without breaking lines
            connections = self._create_pulli_connections(dots)
        elif pattern_type == PatternType.SIKKU_KOLAM:
            # Sikku kolam: continuous line without lifting
            connections = self._create_sikku_connections(dots)
        elif pattern_type == PatternType.MANDALAM:
            # Circular connections from center
            connections = self._create_mandalam_connections(dots)
        
        return connections
    
    def _create_pulli_connections(self, dots: List[Point]) -> List[Tuple[Point, Point]]:
        """Create connections for pulli kolam following traditional rules"""
        connections = []
        # Simple implementation: connect nearby dots
        for i, dot1 in enumerate(dots):
            for j, dot2 in enumerate(dots[i+1:], i+1):
                distance = dot1.distance(dot2)
                if distance < 3.0:  # Connect nearby dots
                    connections.append((dot1, dot2))
        return connections
    
    def _create_sikku_connections(self, dots: List[Point]) -> List[Tuple[Point, Point]]:
        """Create continuous line connections for sikku kolam"""
        if not dots:
            return []
        
        connections = []
        # Create a path that visits all dots
        remaining_dots = dots.copy()
        current_dot = remaining_dots.pop(0)
        
        while remaining_dots:
            # Find nearest unvisited dot
            nearest_dot = min(remaining_dots, key=lambda d: current_dot.distance(d))
            connections.append((current_dot, nearest_dot))
            remaining_dots.remove(nearest_dot)
            current_dot = nearest_dot
        
        return connections
    
    def _create_mandalam_connections(self, dots: List[Point]) -> List[Tuple[Point, Point]]:
        """Create radial connections for mandalam patterns"""
        if not dots:
            return []
        
        # Find center point
        center_x = sum(d.x for d in dots) / len(dots)
        center_y = sum(d.y for d in dots) / len(dots)
        center = Point(center_x, center_y)
        
        connections = []
        # Connect all dots to center
        for dot in dots:
            connections.append((center, dot))
        
        return connections
    
    def calculate_authenticity_score(self, pattern_data: Dict[str, Any]) -> float:
        """Calculate cultural authenticity score (0-1)"""
        score = 0.0
        max_score = 0.0
        
        # Grid size authenticity
        grid_size = pattern_data.get('grid_size', 0)
        max_score += 0.2
        if grid_size in self.rules.PREFERRED_GRID_SIZES:
            score += 0.2
        elif grid_size in self.rules.AUSPICIOUS_NUMBERS:
            score += 0.15
        elif grid_size not in self.rules.FORBIDDEN_NUMBERS:
            score += 0.1
        
        # Symmetry authenticity
        max_score += 0.3
        symmetry_data = pattern_data.get('symmetry', {})
        if any(symmetry_data.get(f'rotational_{order}', False) for order in [4, 6, 8]):
            score += 0.3
        elif symmetry_data.get('rotational_2', False):
            score += 0.2
        elif any(symmetry_data.get(f'reflective_{axes}', False) for axes in [1, 2, 4]):
            score += 0.25
        
        # Pattern complexity
        max_score += 0.2
        num_elements = pattern_data.get('num_elements', 0)
        if 20 <= num_elements <= 100:  # Optimal complexity range
            score += 0.2
        elif 10 <= num_elements <= 150:
            score += 0.15
        elif num_elements > 0:
            score += 0.1
        
        # Connectivity authenticity
        max_score += 0.3
        connectivity_type = pattern_data.get('connectivity_type', '')
        if connectivity_type == 'continuous_single_line':  # Traditional sikku kolam
            score += 0.3
        elif connectivity_type == 'all_connected':  # Traditional pulli kolam
            score += 0.25
        elif connectivity_type == 'radial':  # Traditional mandalam
            score += 0.2
        
        return score / max_score if max_score > 0 else 0.0

class PatternGenerator:
    """Main Kolam pattern generation engine"""
    
    def __init__(self, cultural_rules: CulturalRules = None):
        self.cultural_rules = cultural_rules or CulturalRules()
        self.rule_engine = RuleEngine(self.cultural_rules)
        self.symmetry_controller = SymmetryController()
        
        # Generation parameters
        self.default_params = {
            'grid_size': 9,
            'pattern_type': PatternType.PULLI_KOLAM,
            'symmetry_types': [SymmetryType.ROTATIONAL],
            'complexity_level': 'medium',
            'cultural_theme': 'traditional',
            'color_scheme': 'white_on_red'
        }
    
    def generate(self, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a complete Kolam pattern"""
        # Merge with default parameters
        params = {**self.default_params, **(parameters or {})}
        
        logger.info(f"Generating Kolam pattern with parameters: {params}")
        
        try:
            # Validate parameters
            validation_result = self._validate_parameters(params)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['message'],
                    'pattern_data': None
                }
            
            # Generate base pattern
            base_pattern = self._generate_base_pattern(params)
            
            # Apply symmetry
            symmetric_pattern = self._apply_symmetry(base_pattern, params)
            
            # Apply cultural rules
            refined_pattern = self.rule_engine.apply_connectivity_rules(
                symmetric_pattern['dots'], 
                params['pattern_type']
            )
            
            # Calculate metrics
            pattern_data = {
                'dots': symmetric_pattern['dots'],
                'connections': refined_pattern,
                'grid_size': params['grid_size'],
                'pattern_type': params['pattern_type'].value,
                'symmetry': symmetric_pattern['symmetry_data'],
                'num_elements': len(symmetric_pattern['dots']) + len(refined_pattern),
                'connectivity_type': self._determine_connectivity_type(refined_pattern),
                'parameters': params
            }
            
            # Calculate authenticity score
            authenticity_score = self.rule_engine.calculate_authenticity_score(pattern_data)
            
            # Generate SVG representation
            svg_data = self._generate_svg(pattern_data)
            
            result = {
                'success': True,
                'pattern_data': pattern_data,
                'authenticity_score': authenticity_score,
                'svg_data': svg_data,
                'cultural_info': self._get_cultural_info(params),
                'generation_metadata': {
                    'timestamp': self._get_timestamp(),
                    'parameters_used': params,
                    'validation_passed': True
                }
            }
            
            logger.info(f"Pattern generated successfully. Authenticity score: {authenticity_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating pattern: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'pattern_data': None
            }
    
    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generation parameters"""
        grid_size = params.get('grid_size', 9)
        is_valid, message = self.rule_engine.validate_grid_size(grid_size)
        
        if not is_valid:
            return {'valid': False, 'message': message}
        
        # Validate pattern type
        pattern_type = params.get('pattern_type')
        if not isinstance(pattern_type, PatternType):
            return {'valid': False, 'message': f"Invalid pattern type: {pattern_type}"}
        
        return {'valid': True, 'message': 'Parameters valid'}
    
    def _generate_base_pattern(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate base pattern before applying symmetry"""
        grid_size = params['grid_size']
        pattern_type = params['pattern_type']
        
        if pattern_type == PatternType.PULLI_KOLAM:
            dots = self._generate_pulli_dots(grid_size)
        elif pattern_type == PatternType.SIKKU_KOLAM:
            dots = self._generate_sikku_base(grid_size)
        elif pattern_type == PatternType.MANDALAM:
            dots = self._generate_mandalam_base(grid_size)
        else:
            # Default to pulli kolam
            dots = self._generate_pulli_dots(grid_size)
        
        return {
            'dots': dots,
            'center': Point(grid_size/2, grid_size/2),
            'grid_size': grid_size
        }
    
    def _generate_pulli_dots(self, grid_size: int) -> List[Point]:
        """Generate dots for pulli kolam pattern"""
        dots = []
        center = grid_size // 2
        
        # Create a symmetric dot pattern
        for i in range(grid_size):
            for j in range(grid_size):
                # Skip center and create interesting pattern
                if (i + j) % 2 == 0 and not (i == center and j == center):
                    dots.append(Point(i, j))
        
        return dots
    
    def _generate_sikku_base(self, grid_size: int) -> List[Point]:
        """Generate base points for sikku kolam"""
        dots = []
        center = grid_size // 2
        
        # Create points along the edges for continuous line drawing
        for i in range(0, grid_size, 2):
            dots.append(Point(i, 0))  # Top edge
            dots.append(Point(i, grid_size-1))  # Bottom edge
            dots.append(Point(0, i))  # Left edge
            dots.append(Point(grid_size-1, i))  # Right edge
        
        # Add some interior points
        for i in range(2, grid_size-2, 3):
            for j in range(2, grid_size-2, 3):
                dots.append(Point(i, j))
        
        # Remove duplicates
        unique_dots = []
        for dot in dots:
            if not any(d.distance(dot) < 0.1 for d in unique_dots):
                unique_dots.append(dot)
        
        return unique_dots
    
    def _generate_mandalam_base(self, grid_size: int) -> List[Point]:
        """Generate base points for mandalam pattern"""
        dots = []
        center = Point(grid_size/2, grid_size/2)
        
        # Create concentric circles
        num_circles = 3
        for circle in range(1, num_circles + 1):
            radius = circle * grid_size / (2 * num_circles)
            num_points = 8 * circle  # More points in outer circles
            
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                x = center.x + radius * math.cos(angle)
                y = center.y + radius * math.sin(angle)
                dots.append(Point(x, y))
        
        return dots
    
    def _apply_symmetry(self, base_pattern: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply symmetry transformations to base pattern"""
        dots = base_pattern['dots']
        center = base_pattern['center']
        symmetry_types = params.get('symmetry_types', [SymmetryType.ROTATIONAL])
        
        symmetric_dots = dots.copy()
        
        # Apply each symmetry type
        for symmetry_type in symmetry_types:
            if symmetry_type == SymmetryType.ROTATIONAL:
                symmetric_dots = self.symmetry_controller.apply_rotational_symmetry(
                    symmetric_dots, center, 4
                )
            elif symmetry_type == SymmetryType.REFLECTIVE:
                symmetric_dots = self.symmetry_controller.apply_reflective_symmetry(
                    symmetric_dots, center, 2
                )
            elif symmetry_type == SymmetryType.POINT:
                symmetric_dots = self.symmetry_controller.apply_point_symmetry(
                    symmetric_dots, center
                )
        
        # Remove duplicates
        unique_dots = []
        for dot in symmetric_dots:
            if not any(d.distance(dot) < 0.1 for d in unique_dots):
                unique_dots.append(dot)
        
        # Validate symmetry
        symmetry_validation = self.symmetry_controller.validate_symmetry(unique_dots, center)
        
        return {
            'dots': unique_dots,
            'center': center,
            'symmetry_data': symmetry_validation
        }
    
    def _determine_connectivity_type(self, connections: List[Tuple[Point, Point]]) -> str:
        """Determine the type of connectivity in the pattern"""
        if not connections:
            return 'no_connections'
        
        # Analyze connection pattern
        # This is a simplified analysis
        num_connections = len(connections)
        num_dots = len(set([p for conn in connections for p in conn]))
        
        if num_connections == num_dots - 1:
            return 'continuous_single_line'
        elif num_connections > num_dots:
            return 'all_connected'
        else:
            return 'radial'
    
    def _generate_svg(self, pattern_data: Dict[str, Any]) -> str:
        """Generate SVG representation of the pattern"""
        dots = pattern_data['dots']
        connections = pattern_data['connections']
        grid_size = pattern_data['grid_size']
        
        # SVG dimensions
        width = height = 400
        scale = width / (grid_size + 1)
        
        svg_lines = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{width}" height="{height}" fill="#8B0000" stroke="none"/>',  # Dark red background
        ]
        
        # Draw connections
        for p1, p2 in connections:
            x1, y1 = (p1.x + 0.5) * scale, (p1.y + 0.5) * scale
            x2, y2 = (p2.x + 0.5) * scale, (p2.y + 0.5) * scale
            svg_lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="white" stroke-width="2"/>')
        
        # Draw dots
        for dot in dots:
            x, y = (dot.x + 0.5) * scale, (dot.y + 0.5) * scale
            svg_lines.append(f'<circle cx="{x}" cy="{y}" r="3" fill="white"/>')
        
        svg_lines.append('</svg>')
        return '\n'.join(svg_lines)
    
    def _get_cultural_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get cultural information about the generated pattern"""
        pattern_type = params['pattern_type']
        
        cultural_info = {
            PatternType.PULLI_KOLAM: {
                'name': 'Pulli Kolam',
                'description': 'Traditional dot-based kolam with interconnecting lines',
                'significance': 'Represents the interconnectedness of life',
                'usage': 'Daily morning ritual, especially during festivals'
            },
            PatternType.SIKKU_KOLAM: {
                'name': 'Sikku Kolam', 
                'description': 'Continuous line pattern drawn without lifting the hand',
                'significance': 'Symbolizes the continuity of life and cosmic energy',
                'usage': 'Special occasions and festivals'
            },
            PatternType.MANDALAM: {
                'name': 'Mandalam',
                'description': 'Circular cosmic pattern representing the universe',
                'significance': 'Sacred geometry representing divine order',
                'usage': 'Spiritual ceremonies and temple decorations'
            }
        }
        
        return cultural_info.get(pattern_type, {
            'name': 'Traditional Kolam',
            'description': 'Traditional Tamil floor art',
            'significance': 'Auspicious decoration for prosperity',
            'usage': 'Daily and festive occasions'
        })
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def validate_authenticity(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cultural authenticity of a pattern"""
        authenticity_score = self.rule_engine.calculate_authenticity_score(pattern_data)
        
        # Detailed validation results
        validation_details = {
            'overall_score': authenticity_score,
            'grid_size_valid': pattern_data.get('grid_size', 0) in self.cultural_rules.PREFERRED_GRID_SIZES,
            'symmetry_appropriate': any(pattern_data.get('symmetry', {}).values()),
            'cultural_compliance': authenticity_score > 0.7,
            'recommendations': []
        }
        
        # Generate recommendations
        if authenticity_score < 0.5:
            validation_details['recommendations'].append('Consider using traditional grid sizes (odd numbers)')
        if authenticity_score < 0.7:
            validation_details['recommendations'].append('Enhance symmetry for better cultural authenticity')
        if not validation_details['symmetry_appropriate']:
            validation_details['recommendations'].append('Add rotational or reflective symmetry')
        
        return validation_details

# Example usage and testing functions
def demo_pattern_generation():
    """Demonstrate pattern generation capabilities"""
    print("=== Kolam Pattern Generator Demo ===\n")
    
    # Initialize generator
    generator = Pat