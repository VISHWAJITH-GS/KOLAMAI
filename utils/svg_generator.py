"""
SVG Pattern Generator Utility
Handles the creation of scalable vector graphics patterns with curves, symmetry, and optimization.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Point:
    """Represents a 2D point"""
    x: float
    y: float
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)


@dataclass
class PatternElement:
    """Represents a pattern element with styling"""
    path: str
    stroke_width: float = 1.0
    stroke_color: str = "#000000"
    fill_color: str = "none"
    opacity: float = 1.0


class SVGGenerator:
    """Main class for generating SVG patterns"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.elements: List[PatternElement] = []
        self.defs = {}  # For storing reusable elements like gradients
        
    def create_svg(self, pattern_data: Dict[str, Any]) -> str:
        """
        Creates complete SVG format patterns from pattern data
        
        Args:
            pattern_data: Dictionary containing pattern information including:
                - elements: List of pattern elements
                - background: Background color/gradient
                - symmetry: Symmetry type and parameters
                - style: Visual styling information
        
        Returns:
            Complete SVG as string
        """
        # Create root SVG element
        svg = ET.Element('svg')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('width', str(self.width))
        svg.set('height', str(self.height))
        svg.set('viewBox', f'0 0 {self.width} {self.height}')
        
        # Add definitions section for reusable elements
        if self.defs or pattern_data.get('gradients'):
            defs = ET.SubElement(svg, 'defs')
            self._add_definitions(defs, pattern_data)
        
        # Add background
        if pattern_data.get('background'):
            self._add_background(svg, pattern_data['background'])
        
        # Process pattern elements
        if 'elements' in pattern_data:
            for element_data in pattern_data['elements']:
                element = self._create_pattern_element(element_data)
                if element:
                    self.elements.append(element)
        
        # Apply symmetry transformations
        if pattern_data.get('symmetry'):
            self.apply_symmetry(pattern_data['symmetry'])
        
        # Add curves if specified
        if pattern_data.get('add_curves', False):
            self.add_curves(pattern_data.get('curve_params', {}))
        
        # Optimize paths
        if pattern_data.get('optimize', True):
            self.optimize_paths()
        
        # Add all elements to SVG
        for element in self.elements:
            path_elem = ET.SubElement(svg, 'path')
            path_elem.set('d', element.path)
            path_elem.set('stroke', element.stroke_color)
            path_elem.set('stroke-width', str(element.stroke_width))
            path_elem.set('fill', element.fill_color)
            if element.opacity != 1.0:
                path_elem.set('opacity', str(element.opacity))
        
        # Convert to pretty-printed string
        rough_string = ET.tostring(svg, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")[23:]  # Remove XML declaration
    
    def add_curves(self, curve_params: Dict[str, Any]) -> None:
        """
        Adds curved elements to patterns
        
        Args:
            curve_params: Parameters for curve generation including:
                - curve_type: Type of curves (bezier, arc, spiral)
                - density: Number of curves to add
                - amplitude: Curve amplitude
                - frequency: Curve frequency
        """
        curve_type = curve_params.get('curve_type', 'bezier')
        density = curve_params.get('density', 5)
        amplitude = curve_params.get('amplitude', 50)
        frequency = curve_params.get('frequency', 1)
        
        for i in range(density):
            if curve_type == 'bezier':
                curve = self._create_bezier_curve(i, amplitude, frequency, curve_params)
            elif curve_type == 'arc':
                curve = self._create_arc_curve(i, amplitude, frequency, curve_params)
            elif curve_type == 'spiral':
                curve = self._create_spiral_curve(i, amplitude, frequency, curve_params)
            else:
                continue
            
            if curve:
                self.elements.append(curve)
    
    def apply_symmetry(self, symmetry_config: Dict[str, Any]) -> None:
        """
        Applies symmetrical transformations to pattern elements
        
        Args:
            symmetry_config: Symmetry configuration including:
                - type: Symmetry type (mirror, rotational, kaleidoscope)
                - axis: Symmetry axis for mirroring
                - rotations: Number of rotations for rotational symmetry
                - center: Center point for transformations
        """
        symmetry_type = symmetry_config.get('type', 'mirror')
        original_elements = self.elements.copy()
        
        if symmetry_type == 'mirror':
            self._apply_mirror_symmetry(original_elements, symmetry_config)
        elif symmetry_type == 'rotational':
            self._apply_rotational_symmetry(original_elements, symmetry_config)
        elif symmetry_type == 'kaleidoscope':
            self._apply_kaleidoscope_symmetry(original_elements, symmetry_config)
    
    def optimize_paths(self) -> None:
        """
        Optimizes the drawing paths for better performance and smaller file size
        """
        for i, element in enumerate(self.elements):
            # Remove redundant points
            optimized_path = self._remove_redundant_points(element.path)
            
            # Simplify curves
            optimized_path = self._simplify_curves(optimized_path)
            
            # Merge consecutive moves
            optimized_path = self._merge_consecutive_moves(optimized_path)
            
            self.elements[i] = PatternElement(
                path=optimized_path,
                stroke_width=element.stroke_width,
                stroke_color=element.stroke_color,
                fill_color=element.fill_color,
                opacity=element.opacity
            )
    
    def _add_definitions(self, defs: ET.Element, pattern_data: Dict[str, Any]) -> None:
        """Add gradients and other reusable definitions"""
        gradients = pattern_data.get('gradients', [])
        for gradient in gradients:
            if gradient.get('type') == 'linear':
                grad_elem = ET.SubElement(defs, 'linearGradient')
                grad_elem.set('id', gradient['id'])
                grad_elem.set('x1', str(gradient.get('x1', 0)))
                grad_elem.set('y1', str(gradient.get('y1', 0)))
                grad_elem.set('x2', str(gradient.get('x2', 100)))
                grad_elem.set('y2', str(gradient.get('y2', 0)))
                
                for stop in gradient.get('stops', []):
                    stop_elem = ET.SubElement(grad_elem, 'stop')
                    stop_elem.set('offset', str(stop['offset']))
                    stop_elem.set('stop-color', stop['color'])
    
    def _add_background(self, svg: ET.Element, background: Dict[str, Any]) -> None:
        """Add background rectangle to SVG"""
        bg_rect = ET.SubElement(svg, 'rect')
        bg_rect.set('width', '100%')
        bg_rect.set('height', '100%')
        bg_rect.set('fill', background.get('color', '#ffffff'))
    
    def _create_pattern_element(self, element_data: Dict[str, Any]) -> Optional[PatternElement]:
        """Create a pattern element from data"""
        element_type = element_data.get('type')
        
        if element_type == 'line':
            return self._create_line(element_data)
        elif element_type == 'circle':
            return self._create_circle(element_data)
        elif element_type == 'polygon':
            return self._create_polygon(element_data)
        elif element_type == 'path':
            return PatternElement(
                path=element_data['path'],
                stroke_width=element_data.get('stroke_width', 1.0),
                stroke_color=element_data.get('stroke_color', '#000000'),
                fill_color=element_data.get('fill_color', 'none'),
                opacity=element_data.get('opacity', 1.0)
            )
        return None
    
    def _create_line(self, data: Dict[str, Any]) -> PatternElement:
        """Create a line element"""
        x1, y1 = data.get('start', (0, 0))
        x2, y2 = data.get('end', (100, 100))
        path = f"M {x1} {y1} L {x2} {y2}"
        
        return PatternElement(
            path=path,
            stroke_width=data.get('stroke_width', 1.0),
            stroke_color=data.get('stroke_color', '#000000'),
            fill_color='none'
        )
    
    def _create_circle(self, data: Dict[str, Any]) -> PatternElement:
        """Create a circle element using path"""
        cx, cy = data.get('center', (50, 50))
        r = data.get('radius', 25)
        
        # Create circle using arcs
        path = f"M {cx-r} {cy} A {r} {r} 0 1 0 {cx+r} {cy} A {r} {r} 0 1 0 {cx-r} {cy}"
        
        return PatternElement(
            path=path,
            stroke_width=data.get('stroke_width', 1.0),
            stroke_color=data.get('stroke_color', '#000000'),
            fill_color=data.get('fill_color', 'none')
        )
    
    def _create_polygon(self, data: Dict[str, Any]) -> PatternElement:
        """Create a polygon element"""
        points = data.get('points', [(0,0), (50,0), (25,50)])
        
        path_parts = [f"M {points[0][0]} {points[0][1]}"]
        for point in points[1:]:
            path_parts.append(f"L {point[0]} {point[1]}")
        path_parts.append("Z")
        
        return PatternElement(
            path=" ".join(path_parts),
            stroke_width=data.get('stroke_width', 1.0),
            stroke_color=data.get('stroke_color', '#000000'),
            fill_color=data.get('fill_color', 'none')
        )
    
    def _create_bezier_curve(self, index: int, amplitude: float, frequency: float, 
                           params: Dict[str, Any]) -> PatternElement:
        """Create a Bezier curve"""
        start_x = (index * self.width) / (params.get('density', 5) + 1)
        start_y = self.height / 2
        
        end_x = start_x + self.width / (params.get('density', 5) + 1)
        end_y = start_y + amplitude * math.sin(frequency * index)
        
        # Control points
        cp1_x = start_x + (end_x - start_x) / 3
        cp1_y = start_y - amplitude / 2
        cp2_x = start_x + 2 * (end_x - start_x) / 3
        cp2_y = end_y + amplitude / 2
        
        path = f"M {start_x} {start_y} C {cp1_x} {cp1_y} {cp2_x} {cp2_y} {end_x} {end_y}"
        
        return PatternElement(
            path=path,
            stroke_width=params.get('stroke_width', 1.0),
            stroke_color=params.get('stroke_color', '#000000'),
            fill_color='none'
        )
    
    def _create_arc_curve(self, index: int, amplitude: float, frequency: float,
                         params: Dict[str, Any]) -> PatternElement:
        """Create an arc curve"""
        cx = (index * self.width) / (params.get('density', 5) + 1)
        cy = self.height / 2
        r = amplitude
        
        start_angle = index * frequency * math.pi / 4
        end_angle = start_angle + math.pi
        
        start_x = cx + r * math.cos(start_angle)
        start_y = cy + r * math.sin(start_angle)
        end_x = cx + r * math.cos(end_angle)
        end_y = cy + r * math.sin(end_angle)
        
        large_arc = 1 if abs(end_angle - start_angle) > math.pi else 0
        sweep = 1
        
        path = f"M {start_x} {start_y} A {r} {r} 0 {large_arc} {sweep} {end_x} {end_y}"
        
        return PatternElement(
            path=path,
            stroke_width=params.get('stroke_width', 1.0),
            stroke_color=params.get('stroke_color', '#000000'),
            fill_color='none'
        )
    
    def _create_spiral_curve(self, index: int, amplitude: float, frequency: float,
                           params: Dict[str, Any]) -> PatternElement:
        """Create a spiral curve"""
        cx = self.width / 2
        cy = self.height / 2
        
        path_parts = []
        turns = params.get('turns', 3)
        points_per_turn = 20
        
        for i in range(turns * points_per_turn):
            angle = i * 2 * math.pi / points_per_turn
            radius = amplitude * (i / (turns * points_per_turn))
            
            x = cx + radius * math.cos(angle + index * frequency)
            y = cy + radius * math.sin(angle + index * frequency)
            
            if i == 0:
                path_parts.append(f"M {x} {y}")
            else:
                path_parts.append(f"L {x} {y}")
        
        return PatternElement(
            path=" ".join(path_parts),
            stroke_width=params.get('stroke_width', 1.0),
            stroke_color=params.get('stroke_color', '#000000'),
            fill_color='none'
        )
    
    def _apply_mirror_symmetry(self, elements: List[PatternElement], 
                              config: Dict[str, Any]) -> None:
        """Apply mirror symmetry"""
        axis = config.get('axis', 'vertical')
        
        for element in elements:
            if axis == 'vertical':
                mirrored_path = self._mirror_path_vertical(element.path)
            else:
                mirrored_path = self._mirror_path_horizontal(element.path)
            
            mirrored_element = PatternElement(
                path=mirrored_path,
                stroke_width=element.stroke_width,
                stroke_color=element.stroke_color,
                fill_color=element.fill_color,
                opacity=element.opacity
            )
            self.elements.append(mirrored_element)
    
    def _apply_rotational_symmetry(self, elements: List[PatternElement],
                                  config: Dict[str, Any]) -> None:
        """Apply rotational symmetry"""
        rotations = config.get('rotations', 4)
        center = config.get('center', (self.width/2, self.height/2))
        
        for i in range(1, rotations):
            angle = (2 * math.pi * i) / rotations
            for element in elements:
                rotated_path = self._rotate_path(element.path, angle, center)
                rotated_element = PatternElement(
                    path=rotated_path,
                    stroke_width=element.stroke_width,
                    stroke_color=element.stroke_color,
                    fill_color=element.fill_color,
                    opacity=element.opacity
                )
                self.elements.append(rotated_element)
    
    def _apply_kaleidoscope_symmetry(self, elements: List[PatternElement],
                                   config: Dict[str, Any]) -> None:
        """Apply kaleidoscope symmetry (combination of mirror and rotational)"""
        # First apply rotational symmetry
        self._apply_rotational_symmetry(elements, config)
        # Then apply mirror symmetry to all elements
        current_elements = self.elements.copy()
        self._apply_mirror_symmetry(current_elements, config)
    
    def _mirror_path_vertical(self, path: str) -> str:
        """Mirror a path vertically around the center"""
        # Simple implementation - in practice would need full SVG path parsing
        # This is a simplified version
        return path.replace('M', 'M').replace('L', 'L')  # Placeholder
    
    def _mirror_path_horizontal(self, path: str) -> str:
        """Mirror a path horizontally around the center"""
        # Simple implementation - in practice would need full SVG path parsing
        return path.replace('M', 'M').replace('L', 'L')  # Placeholder
    
    def _rotate_path(self, path: str, angle: float, center: Tuple[float, float]) -> str:
        """Rotate a path around a center point"""
        # Simple implementation - in practice would need full SVG path parsing
        return path  # Placeholder
    
    def _remove_redundant_points(self, path: str) -> str:
        """Remove redundant points from path"""
        # Implementation would parse path and remove consecutive identical points
        return path
    
    def _simplify_curves(self, path: str) -> str:
        """Simplify curves by reducing control points where possible"""
        # Implementation would analyze curves and simplify where appropriate
        return path
    
    def _merge_consecutive_moves(self, path: str) -> str:
        """Merge consecutive move commands"""
        # Implementation would parse path and merge consecutive M commands
        return path


# Usage example
if __name__ == "__main__":
    # Create SVG generator
    generator = SVGGenerator(800, 600)
    
    # Sample pattern data
    pattern_data = {
        'background': {'color': '#f8f8f8'},
        'elements': [
            {
                'type': 'line',
                'start': (100, 100),
                'end': (700, 500),
                'stroke_width': 2,
                'stroke_color': '#2563eb'
            },
            {
                'type': 'circle',
                'center': (400, 300),
                'radius': 50,
                'stroke_width': 3,
                'stroke_color': '#dc2626',
                'fill_color': 'none'
            }
        ],
        'symmetry': {
            'type': 'rotational',
            'rotations': 6,
            'center': (400, 300)
        },
        'add_curves': True,
        'curve_params': {
            'curve_type': 'bezier',
            'density': 3,
            'amplitude': 100,
            'frequency': 2
        }
    }
    
    # Generate SVG
    svg_output = generator.create_svg(pattern_data)
    print(svg_output)