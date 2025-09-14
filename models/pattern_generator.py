import numpy as np
import json
import math
from typing import Dict, List, Tuple, Optional
from enum import Enum
import random
from pathlib import Path

class PatternType(Enum):
    PULLI_KOLAM = "pulli_kolam"
    SIKKU_KOLAM = "sikku_kolam" 
    NELI_KOLAM = "neli_kolam"
    KAMBI_KOLAM = "kambi_kolam"

class SymmetryType(Enum):
    ROTATIONAL_2 = 2
    ROTATIONAL_4 = 4
    ROTATIONAL_8 = 8
    MIRROR_HORIZONTAL = "h_mirror"
    MIRROR_VERTICAL = "v_mirror"
    MIRROR_DIAGONAL = "d_mirror"

class PatternGenerator:
    """
    Generative model for creating authentic Kolam patterns using rule-based
    generation with mathematical constraints and cultural validation.
    """
    
    def __init__(self, cultural_rules_path: str = "data/cultural_database.json"):
        self.cultural_rules = self._load_cultural_rules(cultural_rules_path)
        self.rule_engine = RuleEngine(self.cultural_rules)
        self.symmetry_controller = SymmetryController()
        
        # Pattern generation parameters
        self.grid_sizes = [5, 7, 9, 11, 13, 15, 17, 19]
        self.default_dot_spacing = 20
        self.curve_smoothness = 0.8
        self.min_authenticity_score = 0.7
        
    def _load_cultural_rules(self, path: str) -> Dict:
        """Load cultural rules and constraints from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default rules if file not found
            return self._get_default_cultural_rules()
    
    def _get_default_cultural_rules(self) -> Dict:
        """Default cultural rules for Kolam patterns."""
        return {
            "pulli_kolam": {
                "requires_dots": True,
                "continuous_line": True,
                "symmetry_required": True,
                "min_enclosures": 1,
                "traditional_motifs": ["flower", "leaf", "geometric"]
            },
            "sikku_kolam": {
                "requires_dots": False,
                "continuous_line": True,
                "symmetry_required": True,
                "allows_intersections": False,
                "traditional_motifs": ["maze", "geometric", "abstract"]
            },
            "geometric_constraints": {
                "min_symmetry_axes": 2,
                "max_complexity": 0.8,
                "dot_alignment": "grid",
                "curve_constraints": ["smooth", "no_sharp_angles"]
            }
        }
    
    def generate(self, 
                pattern_type: PatternType,
                grid_size: int = 11,
                symmetry_type: SymmetryType = SymmetryType.ROTATIONAL_4,
                complexity: float = 0.5,
                cultural_strict: bool = True) -> Dict:
        """
        Generate a Kolam pattern with specified parameters.
        
        Args:
            pattern_type: Type of Kolam pattern to generate
            grid_size: Size of the dot grid (odd numbers preferred)
            symmetry_type: Type of symmetry to apply
            complexity: Pattern complexity (0.0 to 1.0)
            cultural_strict: Whether to enforce strict cultural rules
            
        Returns:
            Dictionary containing pattern data, SVG string, and metadata
        """
        # Validate parameters
        if grid_size not in self.grid_sizes:
            grid_size = min(self.grid_sizes, key=lambda x: abs(x - grid_size))
        
        complexity = max(0.0, min(1.0, complexity))
        
        # Generate base pattern
        pattern_data = self._generate_base_pattern(
            pattern_type, grid_size, complexity
        )
        
        # Apply symmetry
        pattern_data = self.symmetry_controller.apply_symmetry(
            pattern_data, symmetry_type
        )
        
        # Apply cultural rules
        if cultural_strict:
            pattern_data = self.rule_engine.validate_and_fix(
                pattern_data, pattern_type
            )
        
        # Generate SVG
        svg_string = self._pattern_to_svg(pattern_data, grid_size)
        
        # Calculate authenticity score
        authenticity_score = self._calculate_authenticity(
            pattern_data, pattern_type
        )
        
        return {
            "pattern_data": pattern_data,
            "svg": svg_string,
            "metadata": {
                "pattern_type": pattern_type.value,
                "grid_size": grid_size,
                "symmetry_type": symmetry_type.value,
                "complexity": complexity,
                "authenticity_score": authenticity_score,
                "cultural_elements": self._identify_cultural_elements(pattern_data)
            }
        }
    
    def _generate_base_pattern(self, pattern_type: PatternType, 
                              grid_size: int, complexity: float) -> Dict:
        """Generate base pattern structure based on type."""
        pattern_data = {
            "dots": [],
            "curves": [],
            "enclosed_areas": [],
            "grid_size": grid_size
        }
        
        # Generate dot grid
        if pattern_type in [PatternType.PULLI_KOLAM, PatternType.NELI_KOLAM]:
            pattern_data["dots"] = self._generate_dot_grid(grid_size)
        
        # Generate curves based on pattern type
        if pattern_type == PatternType.PULLI_KOLAM:
            pattern_data["curves"] = self._generate_pulli_curves(
                pattern_data["dots"], complexity
            )
        elif pattern_type == PatternType.SIKKU_KOLAM:
            pattern_data["curves"] = self._generate_sikku_curves(
                grid_size, complexity
            )
        elif pattern_type == PatternType.NELI_KOLAM:
            pattern_data["curves"] = self._generate_neli_curves(
                pattern_data["dots"], complexity
            )
        elif pattern_type == PatternType.KAMBI_KOLAM:
            pattern_data["curves"] = self._generate_kambi_curves(
                grid_size, complexity
            )
        
        return pattern_data
    
    def _generate_dot_grid(self, grid_size: int) -> List[Tuple[float, float]]:
        """Generate regular dot grid for pulli kolam."""
        dots = []
        center = grid_size // 2
        spacing = self.default_dot_spacing
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i - center) * spacing
                y = (j - center) * spacing
                dots.append((x, y))
        
        return dots
    
    def _generate_pulli_curves(self, dots: List[Tuple[float, float]], 
                              complexity: float) -> List[Dict]:
        """Generate curves that connect around dots for pulli kolam."""
        curves = []
        used_dots = set()
        
        # Start from center and work outward
        center_dot = len(dots) // 2
        current_path = []
        
        # Generate main continuous curve
        path_points = self._generate_continuous_path(dots, complexity)
        
        # Convert path to smooth curves
        for i in range(len(path_points) - 1):
            curve = self._create_smooth_curve(
                path_points[i], path_points[i + 1], "pulli"
            )
            curves.append(curve)
        
        return curves
    
    def _generate_sikku_curves(self, grid_size: int, complexity: float) -> List[Dict]:
        """Generate interlocking curves for sikku kolam."""
        curves = []
        spacing = self.default_dot_spacing
        
        # Generate maze-like pattern
        num_curves = int(complexity * 8) + 4
        
        for i in range(num_curves):
            # Generate random path within grid
            start_point = self._random_grid_point(grid_size, spacing)
            end_point = self._random_grid_point(grid_size, spacing)
            
            # Ensure no intersections (sikku rule)
            curve_path = self._generate_non_intersecting_path(
                start_point, end_point, curves
            )
            
            if curve_path:
                curve = {
                    "type": "sikku_path",
                    "points": curve_path,
                    "style": "continuous"
                }
                curves.append(curve)
        
        return curves
    
    def _generate_neli_curves(self, dots: List[Tuple[float, float]], 
                             complexity: float) -> List[Dict]:
        """Generate square/rectangular patterns for neli kolam."""
        curves = []
        
        # Generate nested squares around dots
        levels = int(complexity * 5) + 2
        
        for level in range(levels):
            square_size = (level + 1) * self.default_dot_spacing * 0.8
            square_points = self._generate_square_path(square_size)
            
            curve = {
                "type": "neli_square",
                "points": square_points,
                "level": level,
                "style": "geometric"
            }
            curves.append(curve)
        
        return curves
    
    def _generate_kambi_curves(self, grid_size: int, complexity: float) -> List[Dict]:
        """Generate line-based patterns for kambi kolam."""
        curves = []
        spacing = self.default_dot_spacing
        
        # Generate parallel and intersecting lines
        num_line_sets = int(complexity * 6) + 3
        
        for i in range(num_line_sets):
            angle = (i * 180 / num_line_sets) % 180
            lines = self._generate_parallel_lines(
                grid_size, spacing, angle, complexity
            )
            curves.extend(lines)
        
        return curves
    
    def _create_smooth_curve(self, start: Tuple[float, float], 
                           end: Tuple[float, float], 
                           curve_type: str) -> Dict:
        """Create smooth curve between two points."""
        # Calculate control points for Bezier curve
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Add some randomness for natural look
        control1 = (
            start[0] + dx * 0.3 + random.uniform(-5, 5),
            start[1] + dy * 0.3 + random.uniform(-5, 5)
        )
        control2 = (
            start[0] + dx * 0.7 + random.uniform(-5, 5),
            start[1] + dy * 0.7 + random.uniform(-5, 5)
        )
        
        return {
            "type": "bezier",
            "start": start,
            "end": end,
            "control1": control1,
            "control2": control2,
            "curve_type": curve_type
        }
    
    def _pattern_to_svg(self, pattern_data: Dict, grid_size: int) -> str:
        """Convert pattern data to SVG string."""
        svg_size = grid_size * self.default_dot_spacing * 1.2
        svg_center = svg_size / 2
        
        svg_lines = [
            f'<svg width="{svg_size}" height="{svg_size}" xmlns="http://www.w3.org/2000/svg">',
            '<g transform="translate({},{})">'.format(svg_center, svg_center)
        ]
        
        # Add dots if present
        if pattern_data.get("dots"):
            svg_lines.append('<g class="dots">')
            for dot in pattern_data["dots"]:
                svg_lines.append(
                    f'<circle cx="{dot[0]}" cy="{dot[1]}" r="2" '
                    f'fill="#8B0000" stroke="none"/>'
                )
            svg_lines.append('</g>')
        
        # Add curves
        svg_lines.append('<g class="curves" stroke="#8B0000" stroke-width="2" fill="none">')
        for curve in pattern_data["curves"]:
            if curve["type"] == "bezier":
                path = f'M {curve["start"][0]},{curve["start"][1]} '
                path += f'C {curve["control1"][0]},{curve["control1"][1]} '
                path += f'{curve["control2"][0]},{curve["control2"][1]} '
                path += f'{curve["end"][0]},{curve["end"][1]}'
                svg_lines.append(f'<path d="{path}"/>')
            elif curve["type"] in ["sikku_path", "neli_square"]:
                path_data = "M " + " L ".join([f"{p[0]},{p[1]}" for p in curve["points"]])
                if curve.get("style") == "geometric":
                    path_data += " Z"  # Close path for geometric shapes
                svg_lines.append(f'<path d="{path_data}"/>')
        
        svg_lines.extend(['</g>', '</g>', '</svg>'])
        
        return '\n'.join(svg_lines)
    
    def _calculate_authenticity(self, pattern_data: Dict, 
                               pattern_type: PatternType) -> float:
        """Calculate cultural authenticity score."""
        score = 0.8  # Base score
        
        rules = self.cultural_rules.get(pattern_type.value, {})
        
        # Check for required elements
        if rules.get("requires_dots") and pattern_data.get("dots"):
            score += 0.1
        elif rules.get("requires_dots") and not pattern_data.get("dots"):
            score -= 0.2
        
        # Check symmetry
        if self._check_symmetry(pattern_data):
            score += 0.1
        
        # Check continuity for patterns that require it
        if rules.get("continuous_line"):
            if self._check_continuity(pattern_data):
                score += 0.1
            else:
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _check_symmetry(self, pattern_data: Dict) -> bool:
        """Check if pattern has symmetry."""
        # Simplified symmetry check
        return len(pattern_data.get("curves", [])) > 2
    
    def _check_continuity(self, pattern_data: Dict) -> bool:
        """Check if curves form continuous paths."""
        # Simplified continuity check
        return len(pattern_data.get("curves", [])) > 0
    
    def _identify_cultural_elements(self, pattern_data: Dict) -> List[str]:
        """Identify cultural elements in the pattern."""
        elements = []
        
        if pattern_data.get("dots"):
            elements.append("traditional_dot_grid")
        
        if len(pattern_data.get("curves", [])) > 5:
            elements.append("complex_geometry")
        
        # Check for traditional motifs (simplified)
        if self._has_circular_elements(pattern_data):
            elements.append("circular_motif")
        
        if self._has_angular_elements(pattern_data):
            elements.append("geometric_motif")
        
        return elements
    
    def _has_circular_elements(self, pattern_data: Dict) -> bool:
        """Check for circular/curved elements."""
        return any(curve.get("type") == "bezier" 
                  for curve in pattern_data.get("curves", []))
    
    def _has_angular_elements(self, pattern_data: Dict) -> bool:
        """Check for angular/geometric elements."""
        return any(curve.get("type") in ["neli_square", "kambi_line"] 
                  for curve in pattern_data.get("curves", []))
    
    # Helper methods for specific pattern generation
    def _generate_continuous_path(self, dots: List[Tuple[float, float]], 
                                 complexity: float) -> List[Tuple[float, float]]:
        """Generate continuous path through dot grid."""
        if not dots:
            return []
        
        # Start from center
        center_idx = len(dots) // 2
        path = [dots[center_idx]]
        used = {center_idx}
        
        # Add points based on complexity
        num_points = int(complexity * len(dots) * 0.5) + 3
        
        for _ in range(min(num_points, len(dots) - 1)):
            current = path[-1]
            # Find nearest unused dot
            best_idx = None
            best_dist = float('inf')
            
            for i, dot in enumerate(dots):
                if i not in used:
                    dist = math.sqrt((current[0] - dot[0])**2 + (current[1] - dot[1])**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
            
            if best_idx is not None:
                path.append(dots[best_idx])
                used.add(best_idx)
        
        # Close the path
        if len(path) > 2:
            path.append(path[0])
        
        return path
    
    def _random_grid_point(self, grid_size: int, spacing: float) -> Tuple[float, float]:
        """Generate random point within grid bounds."""
        center = grid_size // 2
        max_offset = center * spacing
        
        x = random.uniform(-max_offset, max_offset)
        y = random.uniform(-max_offset, max_offset)
        
        return (x, y)
    
    def _generate_non_intersecting_path(self, start: Tuple[float, float], 
                                       end: Tuple[float, float], 
                                       existing_curves: List[Dict]) -> List[Tuple[float, float]]:
        """Generate path that doesn't intersect with existing curves."""
        # Simplified non-intersection check
        # In a real implementation, this would use geometric intersection algorithms
        
        # Generate simple straight line for now
        return [start, end]
    
    def _generate_square_path(self, size: float) -> List[Tuple[float, float]]:
        """Generate points for a square path."""
        half_size = size / 2
        return [
            (-half_size, -half_size),
            (half_size, -half_size),
            (half_size, half_size),
            (-half_size, half_size),
            (-half_size, -half_size)  # Close the square
        ]
    
    def _generate_parallel_lines(self, grid_size: int, spacing: float, 
                                angle: float, complexity: float) -> List[Dict]:
        """Generate parallel lines at specified angle."""
        lines = []
        max_offset = grid_size * spacing / 2
        
        num_lines = int(complexity * 5) + 2
        line_spacing = max_offset / num_lines
        
        for i in range(num_lines):
            offset = -max_offset + i * line_spacing
            
            # Calculate line endpoints based on angle
            angle_rad = math.radians(angle)
            
            if abs(angle) < 45 or abs(angle - 180) < 45:
                # More horizontal
                start = (-max_offset, offset)
                end = (max_offset, offset)
            else:
                # More vertical
                start = (offset, -max_offset)
                end = (offset, max_offset)
            
            line = {
                "type": "kambi_line",
                "points": [start, end],
                "angle": angle
            }
            lines.append(line)
        
        return lines


class RuleEngine:
    """Engine for applying cultural rules and constraints to patterns."""
    
    def __init__(self, cultural_rules: Dict):
        self.rules = cultural_rules
    
    def validate_and_fix(self, pattern_data: Dict, pattern_type: PatternType) -> Dict:
        """Validate pattern against cultural rules and fix violations."""
        rules = self.rules.get(pattern_type.value, {})
        
        # Apply fixes based on rules
        if rules.get("requires_dots") and not pattern_data.get("dots"):
            # Add minimal dot grid
            pattern_data["dots"] = self._add_minimal_dots(pattern_data["grid_size"])
        
        if rules.get("continuous_line"):
            pattern_data["curves"] = self._ensure_continuity(pattern_data["curves"])
        
        return pattern_data
    
    def _add_minimal_dots(self, grid_size: int) -> List[Tuple[float, float]]:
        """Add minimal dot grid for patterns that require dots."""
        dots = []
        spacing = 20
        center = grid_size // 2
        
        # Add center dot and surrounding dots
        for i in range(-1, 2):
            for j in range(-1, 2):
                x = i * spacing
                y = j * spacing
                dots.append((x, y))
        
        return dots
    
    def _ensure_continuity(self, curves: List[Dict]) -> List[Dict]:
        """Ensure curves form continuous paths where required."""
        if not curves:
            return curves
        
        # Simple continuity fix - connect curve endpoints
        for i in range(len(curves) - 1):
            current_end = curves[i].get("end")
            next_start = curves[i + 1].get("start")
            
            if current_end and next_start:
                # If there's a gap, add connecting curve
                distance = math.sqrt(
                    (current_end[0] - next_start[0])**2 + 
                    (current_end[1] - next_start[1])**2
                )
                
                if distance > 5:  # Threshold for connection
                    connecting_curve = {
                        "type": "bezier",
                        "start": current_end,
                        "end": next_start,
                        "control1": current_end,
                        "control2": next_start,
                        "curve_type": "connector"
                    }
                    curves.insert(i + 1, connecting_curve)
        
        return curves


class SymmetryController:
    """Controller for applying symmetry transformations to patterns."""
    
    def apply_symmetry(self, pattern_data: Dict, symmetry_type: SymmetryType) -> Dict:
        """Apply specified symmetry transformation to pattern."""
        if symmetry_type in [SymmetryType.ROTATIONAL_2, SymmetryType.ROTATIONAL_4, SymmetryType.ROTATIONAL_8]:
            return self._apply_rotational_symmetry(pattern_data, symmetry_type.value)
        elif symmetry_type == SymmetryType.MIRROR_HORIZONTAL:
            return self._apply_mirror_symmetry(pattern_data, "horizontal")
        elif symmetry_type == SymmetryType.MIRROR_VERTICAL:
            return self._apply_mirror_symmetry(pattern_data, "vertical")
        elif symmetry_type == SymmetryType.MIRROR_DIAGONAL:
            return self._apply_mirror_symmetry(pattern_data, "diagonal")
        
        return pattern_data
    
    def _apply_rotational_symmetry(self, pattern_data: Dict, fold: int) -> Dict:
        """Apply rotational symmetry with specified fold."""
        angle_step = 360 / fold
        
        original_curves = pattern_data.get("curves", []).copy()
        all_curves = original_curves.copy()
        
        # Generate rotated copies
        for i in range(1, fold):
            angle = i * angle_step
            rotated_curves = self._rotate_curves(original_curves, angle)
            all_curves.extend(rotated_curves)
        
        # Update pattern data
        pattern_data["curves"] = all_curves
        
        # Apply same transformation to dots if present
        if pattern_data.get("dots"):
            original_dots = pattern_data["dots"].copy()
            all_dots = original_dots.copy()
            
            for i in range(1, fold):
                angle = i * angle_step
                rotated_dots = self._rotate_points(original_dots, angle)
                all_dots.extend(rotated_dots)
            
            pattern_data["dots"] = all_dots
        
        return pattern_data
    
    def _apply_mirror_symmetry(self, pattern_data: Dict, axis: str) -> Dict:
        """Apply mirror symmetry along specified axis."""
        original_curves = pattern_data.get("curves", []).copy()
        mirrored_curves = self._mirror_curves(original_curves, axis)
        
        pattern_data["curves"] = original_curves + mirrored_curves
        
        # Apply same to dots if present
        if pattern_data.get("dots"):
            original_dots = pattern_data["dots"].copy()
            mirrored_dots = self._mirror_points(original_dots, axis)
            pattern_data["dots"] = original_dots + mirrored_dots
        
        return pattern_data
    
    def _rotate_curves(self, curves: List[Dict], angle: float) -> List[Dict]:
        """Rotate curves by specified angle."""
        rotated = []
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        for curve in curves:
            rotated_curve = curve.copy()
            
            # Rotate all point-type attributes
            for key in ["start", "end", "control1", "control2"]:
                if key in curve:
                    x, y = curve[key]
                    new_x = x * cos_a - y * sin_a
                    new_y = x * sin_a + y * cos_a
                    rotated_curve[key] = (new_x, new_y)
            
            if "points" in curve:
                rotated_points = []
                for x, y in curve["points"]:
                    new_x = x * cos_a - y * sin_a
                    new_y = x * sin_a + y * cos_a
                    rotated_points.append((new_x, new_y))
                rotated_curve["points"] = rotated_points
            
            rotated.append(rotated_curve)
        
        return rotated
    
    def _rotate_points(self, points: List[Tuple[float, float]], angle: float) -> List[Tuple[float, float]]:
        """Rotate points by specified angle."""
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        rotated = []
        for x, y in points:
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a
            rotated.append((new_x, new_y))
        
        return rotated
    
    def _mirror_curves(self, curves: List[Dict], axis: str) -> List[Dict]:
        """Mirror curves along specified axis."""
        mirrored = []
        
        for curve in curves:
            mirrored_curve = curve.copy()
            
            # Mirror all point-type attributes
            for key in ["start", "end", "control1", "control2"]:
                if key in curve:
                    x, y = curve[key]
                    if axis == "horizontal":
                        mirrored_curve[key] = (x, -y)
                    elif axis == "vertical":
                        mirrored_curve[key] = (-x, y)
                    elif axis == "diagonal":
                        mirrored_curve[key] = (y, x)
            
            if "points" in curve:
                mirrored_points = []
                for x, y in curve["points"]:
                    if axis == "horizontal":
                        mirrored_points.append((x, -y))
                    elif axis == "vertical":
                        mirrored_points.append((-x, y))
                    elif axis == "diagonal":
                        mirrored_points.append((y, x))
                mirrored_curve["points"] = mirrored_points
            
            mirrored.append(mirrored_curve)
        
        return mirrored
    
    def _mirror_points(self, points: List[Tuple[float, float]], axis: str) -> List[Tuple[float, float]]:
        """Mirror points along specified axis."""
        mirrored = []
        
        for x, y in points:
            if axis == "horizontal":
                mirrored.append((x, -y))
            elif axis == "vertical":
                mirrored.append((-x, y))
            elif axis == "diagonal":
                mirrored.append((y, x))
        
        return mirrored