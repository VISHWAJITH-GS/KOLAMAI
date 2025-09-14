"""
SVG Pattern Generation and Export Utilities for Kolam Patterns

This module creates scalable vector graphics for Kolam patterns with precise
geometric constructions, symmetry transformations, and cultural authenticity.
"""

import numpy as np
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

class ColorScheme(Enum):
    """Traditional Kolam color schemes"""
    CLASSIC_WHITE = "classic_white"
    FESTIVAL_COLORS = "festival_colors"
    TAMIL_TRADITIONAL = "tamil_traditional"
    MODERN_VIBRANT = "modern_vibrant"
    MONOCHROME = "monochrome"

class SymmetryType(Enum):
    """Symmetry transformation types"""
    ROTATIONAL = "rotational"
    REFLECTIVE = "reflective"
    TRANSLATIONAL = "translational"
    POINT_REFLECTION = "point_reflection"

@dataclass
class SVGSettings:
    """SVG generation settings"""
    width: int = 800
    height: int = 800
    stroke_width: float = 2.0
    fill_opacity: float = 0.0
    stroke_opacity: float = 1.0
    precision: int = 2
    background_color: str = "#FFFFFF"
    grid_visible: bool = False

@dataclass
class Point:
    """2D point representation"""
    x: float
    y: float
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)
    
    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def rotate(self, angle, center=None):
        """Rotate point around center by angle (radians)"""
        if center is None:
            center = Point(0, 0)
        
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        dx, dy = self.x - center.x, self.y - center.y
        
        return Point(
            center.x + dx * cos_a - dy * sin_a,
            center.y + dx * sin_a + dy * cos_a
        )

class SVGGenerator:
    """Main SVG generation class for Kolam patterns"""
    
    def __init__(self, settings: SVGSettings = None):
        """Initialize SVG generator with settings"""
        self.settings = settings or SVGSettings()
        self.color_schemes = self._initialize_color_schemes()
        self.svg_template = self._create_svg_template()
        self.path_precision = self.settings.precision
        
    def create_svg(self, pattern_data: Dict[str, Any], 
                   color_scheme: ColorScheme = ColorScheme.CLASSIC_WHITE) -> str:
        """
        Create complete SVG from pattern data
        
        Args:
            pattern_data: Pattern information dictionary
            color_scheme: Color scheme to use
            
        Returns:
            Complete SVG string
        """
        # Create root SVG element
        svg_root = self._create_svg_root()
        
        # Add background
        self._add_background(svg_root)
        
        # Add grid if visible
        if self.settings.grid_visible:
            self._add_grid(svg_root, pattern_data)
        
        # Process different pattern elements
        if 'dots' in pattern_data:
            self._add_dots(svg_root, pattern_data['dots'], color_scheme)
        
        if 'lines' in pattern_data:
            self._add_lines(svg_root, pattern_data['lines'], color_scheme)
        
        if 'curves' in pattern_data:
            self._add_curves(svg_root, pattern_data['curves'], color_scheme)
        
        if 'contours' in pattern_data:
            self._add_contours(svg_root, pattern_data['contours'], color_scheme)
        
        # Apply symmetry transformations if specified
        if 'symmetry' in pattern_data:
            self._apply_symmetry_transformations(svg_root, pattern_data['symmetry'])
        
        # Convert to string
        return self._svg_to_string(svg_root)
    
    def add_curves(self, svg_root: ET.Element, curves: List[List[Point]], 
                   color_scheme: ColorScheme = ColorScheme.CLASSIC_WHITE):
        """Add Bezier curves to SVG"""
        colors = self.color_schemes[color_scheme]
        
        for i, curve_points in enumerate(curves):
            if len(curve_points) < 2:
                continue
            
            # Create path element for curve
            path_element = ET.SubElement(svg_root, 'path')
            path_data = self._create_curve_path(curve_points)
            
            path_element.set('d', path_data)
            path_element.set('stroke', colors['primary'])
            path_element.set('stroke-width', str(self.settings.stroke_width))
            path_element.set('fill', 'none')
            path_element.set('stroke-opacity', str(self.settings.stroke_opacity))
            path_element.set('stroke-linecap', 'round')
            path_element.set('stroke-linejoin', 'round')
    
    def apply_symmetry(self, svg_root: ET.Element, symmetry_type: SymmetryType, 
                       order: int = 4, center: Point = None):
        """Apply symmetry transformations to the pattern"""
        if center is None:
            center = Point(self.settings.width / 2, self.settings.height / 2)
        
        # Get all drawable elements
        elements = self._get_drawable_elements(svg_root)
        
        if symmetry_type == SymmetryType.ROTATIONAL:
            self._apply_rotational_symmetry(svg_root, elements, order, center)
        elif symmetry_type == SymmetryType.REFLECTIVE:
            self._apply_reflective_symmetry(svg_root, elements, order, center)
        elif symmetry_type == SymmetryType.POINT_REFLECTION:
            self._apply_point_reflection(svg_root, elements, center)
    
    def optimize_paths(self, svg_string: str) -> str:
        """Optimize SVG paths for smaller file size and better rendering"""
        # Parse SVG
        root = ET.fromstring(svg_string)
        
        # Find all path elements
        paths = root.findall('.//path')
        
        for path in paths:
            if 'd' in path.attrib:
                # Optimize path data
                optimized_data = self._optimize_path_data(path.attrib['d'])
                path.set('d', optimized_data)
        
        return ET.tostring(root, encoding='unicode')
    
    def generate_pulli_kolam(self, grid_size: Tuple[int, int], dot_spacing: float = 30,
                           connection_pattern: str = 'sikku') -> str:
        """Generate traditional Pulli (dot-based) Kolam pattern"""
        rows, cols = grid_size
        center_x = self.settings.width / 2
        center_y = self.settings.height / 2
        
        # Calculate grid offset to center the pattern
        total_width = (cols - 1) * dot_spacing
        total_height = (rows - 1) * dot_spacing
        start_x = center_x - total_width / 2
        start_y = center_y - total_height / 2
        
        # Generate dot grid
        dots = []
        for i in range(rows):
            for j in range(cols):
                x = start_x + j * dot_spacing
                y = start_y + i * dot_spacing
                dots.append(Point(x, y))
        
        # Generate connection lines based on pattern
        lines = self._generate_connection_lines(dots, rows, cols, connection_pattern)
        
        # Create SVG
        pattern_data = {
            'dots': [(dot.x, dot.y) for dot in dots],
            'lines': [[(point.x, point.y) for point in line] for line in lines],
            'symmetry': {'type': 'rotational', 'order': 4}
        }
        
        return self.create_svg(pattern_data)
    
    def generate_sikku_kolam(self, complexity_level: int = 3, 
                           symmetry_order: int = 4) -> str:
        """Generate traditional Sikku (line-based) Kolam pattern"""
        center = Point(self.settings.width / 2, self.settings.height / 2)
        
        # Generate base curves
        curves = self._generate_sikku_curves(center, complexity_level, symmetry_order)
        
        # Create pattern data
        pattern_data = {
            'curves': [[(point.x, point.y) for point in curve] for curve in curves],
            'symmetry': {'type': 'rotational', 'order': symmetry_order}
        }
        
        return self.create_svg(pattern_data, ColorScheme.TAMIL_TRADITIONAL)
    
    def generate_geometric_kolam(self, shape_type: str = 'lotus', 
                               layers: int = 3) -> str:
        """Generate geometric Kolam with traditional shapes"""
        center = Point(self.settings.width / 2, self.settings.height / 2)
        
        if shape_type == 'lotus':
            curves = self._generate_lotus_pattern(center, layers)
        elif shape_type == 'star':
            curves = self._generate_star_pattern(center, layers)
        elif shape_type == 'mandala':
            curves = self._generate_mandala_pattern(center, layers)
        else:
            curves = self._generate_default_geometric(center, layers)
        
        pattern_data = {
            'curves': [[(point.x, point.y) for point in curve] for curve in curves],
            'symmetry': {'type': 'rotational', 'order': 8}
        }
        
        return self.create_svg(pattern_data, ColorScheme.FESTIVAL_COLORS)
    
    def export_print_ready(self, svg_string: str, size_mm: Tuple[float, float] = (210, 297)) -> str:
        """Convert SVG to print-ready format with proper dimensions"""
        width_mm, height_mm = size_mm
        
        # Parse existing SVG
        root = ET.fromstring(svg_string)
        
        # Update dimensions for print
        root.set('width', f'{width_mm}mm')
        root.set('height', f'{height_mm}mm')
        root.set('viewBox', f'0 0 {width_mm} {height_mm}')
        
        # Add print-specific attributes
        root.set('xmlns:inkscape', 'http://www.inkscape.org/namespaces/inkscape')
        root.set('xmlns:sodipodi', 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd')
        
        # Scale content to fit print dimensions
        scale_x = width_mm / self.settings.width
        scale_y = height_mm / self.settings.height
        scale = min(scale_x, scale_y)  # Maintain aspect ratio
        
        # Add scaling transform to content
        g_element = ET.SubElement(root, 'g')
        g_element.set('transform', f'scale({scale})')
        
        # Move all existing content into the group
        content_elements = list(root)
        root.clear()
        root.append(g_element)
        for element in content_elements:
            if element.tag != 'g':
                g_element.append(element)
        
        return ET.tostring(root, encoding='unicode')
    
    # Private helper methods
    def _initialize_color_schemes(self) -> Dict[ColorScheme, Dict[str, str]]:
        """Initialize color scheme definitions"""
        return {
            ColorScheme.CLASSIC_WHITE: {
                'primary': '#FFFFFF',
                'secondary': '#F0F0F0',
                'accent': '#E0E0E0',
                'background': '#000000'
            },
            ColorScheme.FESTIVAL_COLORS: {
                'primary': '#FF6B35',    # Orange-red
                'secondary': '#F7931E',   # Orange
                'accent': '#FFD23F',      # Yellow
                'background': '#FFFFFF'
            },
            ColorScheme.TAMIL_TRADITIONAL: {
                'primary': '#DC143C',     # Crimson
                'secondary': '#FFD700',   # Gold
                'accent': '#FFFFFF',      # White
                'background': '#FFF8DC'   # Cornsilk
            },
            ColorScheme.MODERN_VIBRANT: {
                'primary': '#9B59B6',     # Purple
                'secondary': '#3498DB',   # Blue
                'accent': '#E74C3C',      # Red
                'background': '#2C3E50'   # Dark blue-gray
            },
            ColorScheme.MONOCHROME: {
                'primary': '#2C3E50',     # Dark gray
                'secondary': '#34495E',   # Medium gray
                'accent': '#7F8C8D',      # Light gray
                'background': '#FFFFFF'
            }
        }
    
    def _create_svg_template(self) -> str:
        """Create base SVG template"""
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     width="{self.settings.width}" 
     height="{self.settings.height}" 
     viewBox="0 0 {self.settings.width} {self.settings.height}">
</svg>'''
    
    def _create_svg_root(self) -> ET.Element:
        """Create SVG root element"""
        svg = ET.Element('svg')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('width', str(self.settings.width))
        svg.set('height', str(self.settings.height))
        svg.set('viewBox', f'0 0 {self.settings.width} {self.settings.height}')
        return svg
    
    def _add_background(self, svg_root: ET.Element):
        """Add background rectangle"""
        bg = ET.SubElement(svg_root, 'rect')
        bg.set('width', '100%')
        bg.set('height', '100%')
        bg.set('fill', self.settings.background_color)
    
    def _add_grid(self, svg_root: ET.Element, pattern_data: Dict):
        """Add construction grid"""
        if 'grid_spacing' not in pattern_data:
            return
        
        spacing = pattern_data['grid_spacing']
        
        # Vertical lines
        for x in range(0, self.settings.width + 1, spacing):
            line = ET.SubElement(svg_root, 'line')
            line.set('x1', str(x))
            line.set('y1', '0')
            line.set('x2', str(x))
            line.set('y2', str(self.settings.height))
            line.set('stroke', '#E0E0E0')
            line.set('stroke-width', '0.5')
        
        # Horizontal lines
        for y in range(0, self.settings.height + 1, spacing):
            line = ET.SubElement(svg_root, 'line')
            line.set('x1', '0')
            line.set('y1', str(y))
            line.set('x2', str(self.settings.width))
            line.set('y2', str(y))
            line.set('stroke', '#E0E0E0')
            line.set('stroke-width', '0.5')
    
    def _add_dots(self, svg_root: ET.Element, dots: List[Tuple[float, float]], 
                  color_scheme: ColorScheme):
        """Add dots to SVG"""
        colors = self.color_schemes[color_scheme]
        dot_radius = self.settings.stroke_width
        
        for x, y in dots:
            circle = ET.SubElement(svg_root, 'circle')
            circle.set('cx', str(round(x, self.path_precision)))
            circle.set('cy', str(round(y, self.path_precision)))
            circle.set('r', str(dot_radius))
            circle.set('fill', colors['primary'])
            circle.set('fill-opacity', str(self.settings.stroke_opacity))
    
    def _add_lines(self, svg_root: ET.Element, lines: List[List[Tuple[float, float]]], 
                   color_scheme: ColorScheme):
        """Add lines to SVG"""
        colors = self.color_schemes[color_scheme]
        
        for line_points in lines:
            if len(line_points) < 2:
                continue
            
            path = ET.SubElement(svg_root, 'path')
            path_data = self._create_line_path(line_points)
            
            path.set('d', path_data)
            path.set('stroke', colors['primary'])
            path.set('stroke-width', str(self.settings.stroke_width))
            path.set('fill', 'none')
            path.set('stroke-opacity', str(self.settings.stroke_opacity))
            path.set('stroke-linecap', 'round')
            path.set('stroke-linejoin', 'round')
    
    def _add_curves(self, svg_root: ET.Element, curves: List[List[Tuple[float, float]]], 
                    color_scheme: ColorScheme):
        """Add curves to SVG"""
        colors = self.color_schemes[color_scheme]
        
        for curve_points in curves:
            if len(curve_points) < 2:
                continue
            
            path = ET.SubElement(svg_root, 'path')
            path_data = self._create_smooth_curve_path(curve_points)
            
            path.set('d', path_data)
            path.set('stroke', colors['primary'])
            path.set('stroke-width', str(self.settings.stroke_width))
            path.set('fill', 'none')
            path.set('stroke-opacity', str(self.settings.stroke_opacity))
            path.set('stroke-linecap', 'round')
            path.set('stroke-linejoin', 'round')
    
    def _add_contours(self, svg_root: ET.Element, contours: List, 
                      color_scheme: ColorScheme):
        """Add contours as closed paths"""
        colors = self.color_schemes[color_scheme]
        
        for contour in contours:
            if len(contour) < 3:
                continue
            
            # Convert contour to points
            points = [(float(pt[0][0]), float(pt[0][1])) for pt in contour]
            
            path = ET.SubElement(svg_root, 'path')
            path_data = self._create_closed_path(points)
            
            path.set('d', path_data)
            path.set('stroke', colors['primary'])
            path.set('stroke-width', str(self.settings.stroke_width))
            path.set('fill', colors['secondary'])
            path.set('fill-opacity', str(self.settings.fill_opacity))
            path.set('stroke-opacity', str(self.settings.stroke_opacity))
    
    def _create_line_path(self, points: List[Tuple[float, float]]) -> str:
        """Create SVG path data for straight lines"""
        if not points:
            return ""
        
        path_data = f"M {points[0][0]:.{self.path_precision}f},{points[0][1]:.{self.path_precision}f}"
        
        for x, y in points[1:]:
            path_data += f" L {x:.{self.path_precision}f},{y:.{self.path_precision}f}"
        
        return path_data
    
    def _create_curve_path(self, points: List[Point]) -> str:
        """Create SVG path data for Bezier curves"""
        if len(points) < 2:
            return ""
        
        path_data = f"M {points[0].x:.{self.path_precision}f},{points[0].y:.{self.path_precision}f}"
        
        # Create smooth curves using quadratic Bezier
        for i in range(1, len(points)):
            if i == len(points) - 1:
                # Last point - straight line
                path_data += f" L {points[i].x:.{self.path_precision}f},{points[i].y:.{self.path_precision}f}"
            else:
                # Control point is midway to next point
                next_point = points[i + 1]
                control_x = (points[i].x + next_point.x) / 2
                control_y = (points[i].y + next_point.y) / 2
                
                path_data += f" Q {points[i].x:.{self.path_precision}f},{points[i].y:.{self.path_precision}f} {control_x:.{self.path_precision}f},{control_y:.{self.path_precision}f}"
        
        return path_data
    
    def _create_smooth_curve_path(self, points: List[Tuple[float, float]]) -> str:
        """Create smooth curve using cubic Bezier"""
        if len(points) < 2:
            return ""
        
        path_data = f"M {points[0][0]:.{self.path_precision}f},{points[0][1]:.{self.path_precision}f}"
        
        if len(points) == 2:
            path_data += f" L {points[1][0]:.{self.path_precision}f},{points[1][1]:.{self.path_precision}f}"
            return path_data
        
        # Create smooth curve using cubic Bezier splines
        for i in range(1, len(points) - 1):
            p0 = np.array(points[i - 1])
            p1 = np.array(points[i])
            p2 = np.array(points[i + 1])
            
            # Calculate control points
            cp1 = p1 - (p2 - p0) * 0.15
            cp2 = p1 + (p2 - p0) * 0.15
            
            if i == 1:
                # First curve
                path_data += f" C {cp1[0]:.{self.path_precision}f},{cp1[1]:.{self.path_precision}f} {cp2[0]:.{self.path_precision}f},{cp2[1]:.{self.path_precision}f} {p2[0]:.{self.path_precision}f},{p2[1]:.{self.path_precision}f}"
            else:
                # Subsequent curves using smooth continuation
                path_data += f" S {cp2[0]:.{self.path_precision}f},{cp2[1]:.{self.path_precision}f} {p2[0]:.{self.path_precision}f},{p2[1]:.{self.path_precision}f}"
        
        return path_data
    
    def _create_closed_path(self, points: List[Tuple[float, float]]) -> str:
        """Create closed SVG path"""
        path_data = self._create_line_path(points)
        if path_data:
            path_data += " Z"  # Close path
        return path_data
    
    def _generate_connection_lines(self, dots: List[Point], rows: int, cols: int, 
                                 pattern: str) -> List[List[Point]]:
        """Generate connection lines between dots for Pulli Kolam"""
        lines = []
        
        if pattern == 'sikku':
            # Traditional Sikku pattern - continuous lines around dots
            for i in range(rows - 1):
                for j in range(cols - 1):
                    # Create loop around 2x2 dot grid
                    base_idx = i * cols + j
                    
                    if base_idx + cols + 1 < len(dots):
                        loop_points = [
                            dots[base_idx],
                            Point(dots[base_idx].x, dots[base_idx + cols].y),
                            dots[base_idx + cols + 1],
                            Point(dots[base_idx + 1].x, dots[base_idx].y),
                            dots[base_idx]  # Close the loop
                        ]
                        lines.append(loop_points)
        
        elif pattern == 'simple':
            # Simple connecting lines
            # Horizontal lines
            for i in range(rows):
                row_points = []
                for j in range(cols):
                    row_points.append(dots[i * cols + j])
                if len(row_points) > 1:
                    lines.append(row_points)
            
            # Vertical lines
            for j in range(cols):
                col_points = []
                for i in range(rows):
                    col_points.append(dots[i * cols + j])
                if len(col_points) > 1:
                    lines.append(col_points)
        
        return lines
    
    def _generate_sikku_curves(self, center: Point, complexity: int, 
                             symmetry_order: int) -> List[List[Point]]:
        """Generate Sikku Kolam curves"""
        curves = []
        radius_step = 30
        
        for layer in range(complexity):
            radius = (layer + 1) * radius_step
            curve_points = []
            
            # Generate points for this layer
            points_per_layer = 8 * (layer + 1)
            for i in range(points_per_layer):
                angle = (2 * math.pi * i) / points_per_layer
                
                # Add some variation for organic look
                r_variation = radius + 10 * math.sin(angle * symmetry_order)
                
                x = center.x + r_variation * math.cos(angle)
                y = center.y + r_variation * math.sin(angle)
                curve_points.append(Point(x, y))
            
            # Close the curve
            if curve_points:
                curve_points.append(curve_points[0])
                curves.append(curve_points)
        
        return curves
    
    def _generate_lotus_pattern(self, center: Point, layers: int) -> List[List[Point]]:
        """Generate lotus petal pattern"""
        curves = []
        
        for layer in range(layers):
            petals = 8 + layer * 4  # More petals in outer layers
            radius = 50 + layer * 40
            
            for petal in range(petals):
                angle_start = (2 * math.pi * petal) / petals
                angle_end = (2 * math.pi * (petal + 0.8)) / petals  # Overlap slightly
                
                petal_points = []
                
                # Create petal shape
                for t in np.linspace(0, 1, 20):
                    angle = angle_start + t * (angle_end - angle_start)
                    r = radius * (0.3 + 0.7 * math.sin(t * math.pi))  # Petal shape
                    
                    x = center.x + r * math.cos(angle)
                    y = center.y + r * math.sin(angle)
                    petal_points.append(Point(x, y))
                
                curves.append(petal_points)
        
        return curves
    
    def _generate_star_pattern(self, center: Point, layers: int) -> List[List[Point]]:
        """Generate star pattern"""
        curves = []
        
        for layer in range(layers):
            points = 8  # 8-pointed star
            outer_radius = 60 + layer * 50
            inner_radius = outer_radius * 0.4
            
            star_points = []
            
            for i in range(points * 2):
                angle = (2 * math.pi * i) / (points * 2)
                radius = outer_radius if i % 2 == 0 else inner_radius
                
                x = center.x + radius * math.cos(angle)
                y = center.y + radius * math.sin(angle)
                star_points.append(Point(x, y))
            
            # Close the star
            star_points.append(star_points[0])
            curves.append(star_points)
        
        return curves
    
    def _generate_mandala_pattern(self, center: Point, layers: int) -> List[List[Point]]:
        """Generate mandala pattern"""
        curves = []
        
        for layer in range(layers):
            radius = 40 + layer * 30
            segments = 12 + layer * 6
            
            # Outer circle
            circle_points = []
            for i in range(segments + 1):
                angle = (2 * math.pi * i) / segments
                x = center.x + radius * math.cos(angle)
                y = center.y + radius * math.sin(angle)
                circle_points.append(Point(x, y))
            
            curves.append(circle_points)
            
            # Radial lines
            for i in range(0, segments, 2):
                angle = (2 * math.pi * i) / segments
                line_points = [
                    center,
                    Point(center.x + radius * math.cos(angle),
                          center.y + radius * math.sin(angle))
                ]
                curves.append(line_points)
        
        return curves
    
    def _generate_default_geometric(self, center: Point, layers: int) -> List[List[Point]]:
        """Generate default geometric pattern"""
        curves = []
        
        for layer in range(layers):
            radius = 30 + layer * 25
            sides = 6 + layer  # Hexagon, heptagon, octagon...
            
            polygon_points = []
            for i in range(sides + 1):
                angle = (2 * math.pi * i) / sides
                x = center.x + radius * math.cos(angle)
                y = center.y + radius * math.sin(angle)
                polygon_points.append(Point(x, y))
            
            curves.append(polygon_points)
        
        return curves
    
    def _apply_rotational_symmetry(self, svg_root: ET.Element, elements: List[ET.Element], 
                                 order: int, center: Point):
        """Apply rotational symmetry"""
        angle_step = 2 * math.pi / order
        
        for i in range(1, order):  # Skip first (original)
            angle = i * angle_step
            angle_deg = math.degrees(angle)
            
            group = ET.SubElement(svg_root, 'g')
            group.set('transform', f'rotate({angle_deg:.2f} {center.x:.2f} {center.y:.2f})')
            
            # Copy all elements to the rotated group
            for element in elements:
                group.append(self._copy_element(element))
    
    def _apply_reflective_symmetry(self, svg_root: ET.Element, elements: List[ET.Element], 
                                 axes: int, center: Point):
        """Apply reflective symmetry"""
        angle_step = 180 / axes if axes > 1 else 0
        
        for i in range(axes):
            angle = i * angle_step
            
            group = ET.SubElement(svg_root, 'g')
            
            # Create reflection transformation
            if angle == 0:
                # Vertical reflection
                transform = f'scale(-1, 1) translate(-{2 * center.x:.2f}, 0)'
            else:
                # Reflection across angled axis
                transform = f'rotate({angle:.2f} {center.x:.2f} {center.y:.2f}) scale(-1, 1) translate(-{2 * center.x:.2f}, 0) rotate(-{angle:.2f} {center.x:.2f} {center.y:.2f})'
            
            group.set('transform', transform)
            
            for element in elements:
                group.append(self._copy_element(element))
    
    def _apply_point_reflection(self, svg_root: ET.Element, elements: List[ET.Element], 
                              center: Point):
        """Apply point reflection (180-degree rotation)"""
        group = ET.SubElement(svg_root, 'g')
        group.set('transform', f'rotate(180 {center.x:.2f} {center.y:.2f})')
        
        for element in elements:
            group.append(self._copy_element(element))
    
    def _apply_symmetry_transformations(self, svg_root: ET.Element, symmetry_info: Dict):
        """Apply symmetry transformations based on pattern data"""
        symmetry_type = symmetry_info.get('type', 'rotational')
        order = symmetry_info.get('order', 4)
        center = Point(self.settings.width / 2, self.settings.height / 2)
        
        # Get elements to transform (exclude background)
        elements = self._get_drawable_elements(svg_root)
        
        if symmetry_type == 'rotational':
            self._apply_rotational_symmetry(svg_root, elements, order, center)
        elif symmetry_type == 'reflective':
            self._apply_reflective_symmetry(svg_root, elements, order, center)
        elif symmetry_type == 'point_reflection':
            self._apply_point_reflection(svg_root, elements, center)
    
    def _get_drawable_elements(self, svg_root: ET.Element) -> List[ET.Element]:
        """Get all drawable elements (excluding background and grid)"""
        elements = []
        
        for child in svg_root:
            # Skip background rect and grid elements
            if child.tag == 'rect' and child.get('width') == '100%':
                continue
            if child.get('stroke') == '#E0E0E0':  # Grid color
                continue
            
            elements.append(child)
        
        return elements
    
    def _copy_element(self, element: ET.Element) -> ET.Element:
        """Create a deep copy of an XML element"""
        new_element = ET.Element(element.tag, element.attrib)
        new_element.text = element.text
        new_element.tail = element.tail
        
        for child in element:
            new_element.append(self._copy_element(child))
        
        return new_element
    
    def _optimize_path_data(self, path_data: str) -> str:
        """Optimize SVG path data for smaller size"""
        # Remove unnecessary spaces and precision
        optimized = path_data
        
        # Round coordinates to reduce precision
        import re
        
        def round_match(match):
            try:
                value = float(match.group(0))
                return f"{value:.{self.path_precision}f}"
            except:
                return match.group(0)
        
        # Find all numbers and round them
        number_pattern = r'-?\d+\.?\d*'
        optimized = re.sub(number_pattern, round_match, optimized)
        
        # Remove redundant spaces
        optimized = re.sub(r'\s+', ' ', optimized).strip()
        
        # Convert absolute coordinates to relative where beneficial
        # This is a simplified optimization - full implementation would be more complex
        
        return optimized
    
    def _svg_to_string(self, svg_root: ET.Element) -> str:
        """Convert SVG element tree to formatted string"""
        # Create XML declaration and format nicely
        rough_string = ET.tostring(svg_root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ").replace('<?xml version="1.0" ?>\n', '<?xml version="1.0" encoding="UTF-8"?>\n')
    
    def create_drawing_instructions(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate step-by-step drawing instructions for manual creation"""
        instructions = {
            'title': 'Kolam Drawing Instructions',
            'materials': [
                'Rice flour or kolam powder',
                'Clean, dry surface',
                'Optional: stencils or guides'
            ],
            'steps': []
        }
        
        step_counter = 1
        
        # Dot placement instructions
        if 'dots' in pattern_data:
            dots = pattern_data['dots']
            if dots:
                grid_info = self._analyze_dot_grid(dots)
                instructions['steps'].append({
                    'step': step_counter,
                    'action': 'Place dots',
                    'description': f'Create a {grid_info["rows"]}x{grid_info["cols"]} grid of dots with {grid_info["spacing"]:.1f}cm spacing',
                    'tips': ['Use consistent spacing', 'Start from center and work outward', 'Keep dots small and even']
                })
                step_counter += 1
        
        # Line drawing instructions
        if 'lines' in pattern_data:
            lines = pattern_data['lines']
            for i, line in enumerate(lines):
                if len(line) >= 2:
                    instructions['steps'].append({
                        'step': step_counter,
                        'action': f'Draw line {i + 1}',
                        'description': f'Connect points with smooth, continuous line ({len(line)} segments)',
                        'tips': ['Keep lines flowing and smooth', 'Maintain consistent thickness']
                    })
                    step_counter += 1
        
        # Curve instructions
        if 'curves' in pattern_data:
            curves = pattern_data['curves']
            for i, curve in enumerate(curves):
                instructions['steps'].append({
                    'step': step_counter,
                    'action': f'Draw curve {i + 1}',
                    'description': 'Create smooth curved line following the pattern',
                    'tips': ['Use fluid wrist motion', 'Practice the curve motion first']
                })
                step_counter += 1
        
        # Symmetry instructions
        if 'symmetry' in pattern_data:
            symmetry = pattern_data['symmetry']
            instructions['steps'].append({
                'step': step_counter,
                'action': 'Apply symmetry',
                'description': f'Repeat pattern with {symmetry.get("order", 4)}-fold {symmetry.get("type", "rotational")} symmetry',
                'tips': ['Mark center point clearly', 'Use light guidelines', 'Check alignment frequently']
            })
            step_counter += 1
        
        # Finishing instructions
        instructions['steps'].append({
            'step': step_counter,
            'action': 'Final touches',
            'description': 'Review pattern for completeness and make any necessary touch-ups',
            'tips': ['Fill in any gaps', 'Ensure all lines connect properly', 'Clean up excess powder']
        })
        
        instructions['cultural_notes'] = [
            'Kolam is traditionally drawn at dawn',
            'Patterns are created without lifting the hand (for Sikku Kolam)',
            'Each design has symbolic meaning and purpose',
            'Practice and patience lead to beautiful results'
        ]
        
        return instructions
    
    def _analyze_dot_grid(self, dots: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze dot arrangement to provide grid information"""
        if len(dots) < 2:
            return {'rows': 1, 'cols': 1, 'spacing': 0}
        
        # Find unique x and y coordinates
        x_coords = sorted(list(set(x for x, y in dots)))
        y_coords = sorted(list(set(y for x, y in dots)))
        
        rows = len(y_coords)
        cols = len(x_coords)
        
        # Calculate average spacing
        if len(x_coords) > 1:
            x_spacing = np.mean(np.diff(x_coords))
        else:
            x_spacing = 0
        
        if len(y_coords) > 1:
            y_spacing = np.mean(np.diff(y_coords))
        else:
            y_spacing = 0
        
        # Convert pixels to approximate cm (assuming 72 DPI)
        pixels_per_cm = 72 / 2.54
        avg_spacing_cm = ((x_spacing + y_spacing) / 2) / pixels_per_cm
        
        return {
            'rows': rows,
            'cols': cols,
            'spacing': avg_spacing_cm,
            'x_spacing': x_spacing,
            'y_spacing': y_spacing
        }
    
    def generate_coloring_version(self, svg_string: str) -> str:
        """Generate a coloring book version of the pattern"""
        # Parse SVG
        root = ET.fromstring(svg_string)
        
        # Modify all paths and shapes for coloring
        for element in root.iter():
            if element.tag in ['path', 'circle', 'rect', 'polygon']:
                # Set coloring book properties
                element.set('fill', 'none')
                element.set('stroke', '#000000')
                element.set('stroke-width', '2')
                element.set('stroke-linecap', 'round')
                element.set('stroke-linejoin', 'round')
            
            # Remove any existing fills and colors
            if 'fill' in element.attrib and element.attrib['fill'] != 'none':
                element.set('fill', 'none')
        
        # Set white background
        background = root.find('rect[@width="100%"]')
        if background is not None:
            background.set('fill', '#FFFFFF')
        
        return ET.tostring(root, encoding='unicode')
    
    def create_template_library(self) -> Dict[str, str]:
        """Create a library of common Kolam templates"""
        templates = {}
        
        # Simple Pulli Kolam (5x5 grid)
        templates['simple_pulli'] = self.generate_pulli_kolam(
            (5, 5), 
            dot_spacing=40, 
            connection_pattern='simple'
        )
        
        # Traditional Sikku Kolam
        templates['traditional_sikku'] = self.generate_sikku_kolam(
            complexity_level=2, 
            symmetry_order=4
        )
        
        # Lotus Pattern
        templates['lotus_pattern'] = self.generate_geometric_kolam(
            shape_type='lotus', 
            layers=3
        )
        
        # Star Pattern
        templates['star_pattern'] = self.generate_geometric_kolam(
            shape_type='star', 
            layers=2
        )
        
        # Mandala Pattern
        templates['mandala_pattern'] = self.generate_geometric_kolam(
            shape_type='mandala', 
            layers=4
        )
        
        # Complex Pulli Kolam
        templates['complex_pulli'] = self.generate_pulli_kolam(
            (9, 9), 
            dot_spacing=25, 
            connection_pattern='sikku'
        )
        
        return templates
    
    def batch_export(self, patterns: Dict[str, str], output_format: str = 'svg') -> Dict[str, str]:
        """Batch export multiple patterns in specified format"""
        exported = {}
        
        for name, svg_content in patterns.items():
            if output_format == 'print':
                exported[name] = self.export_print_ready(svg_content)
            elif output_format == 'coloring':
                exported[name] = self.generate_coloring_version(svg_content)
            elif output_format == 'optimized':
                exported[name] = self.optimize_paths(svg_content)
            else:
                exported[name] = svg_content
        
        return exported