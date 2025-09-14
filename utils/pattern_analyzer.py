import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
import math
from scipy import ndimage, spatial
from scipy.stats import entropy
from skimage import measure, morphology
from skimage.feature import graycomatrix, graycoprops
import networkx as nx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """
    Advanced pattern analysis algorithms for Kolam pattern classification and understanding.
    Provides topology analysis, complexity measurement, pattern type detection, and cultural metrics.
    """
    
    def __init__(self):
        # Pattern complexity weights
        self.complexity_weights = {
            'geometric': 0.25,
            'topological': 0.20,
            'symmetry': 0.15,
            'fractal': 0.15,
            'structural': 0.15,
            'density': 0.10
        }
        
        # Pattern signatures for different Kolam types
        self.pattern_signatures = {
            'pulli_kolam': {
                'dot_density_range': (0.05, 0.20),
                'symmetry_threshold': 0.7,
                'line_curve_ratio': (0.3, 0.8),
                'complexity_range': (0.4, 0.8)
            },
            'sikku_kolam': {
                'dot_density_range': (0.02, 0.10),
                'symmetry_threshold': 0.8,
                'line_curve_ratio': (0.6, 1.0),
                'complexity_range': (0.6, 0.9)
            },
            'rangoli': {
                'dot_density_range': (0.15, 0.40),
                'symmetry_threshold': 0.6,
                'line_curve_ratio': (0.2, 0.7),
                'complexity_range': (0.3, 0.7)
            },
            'kambi_kolam': {
                'dot_density_range': (0.01, 0.08),
                'symmetry_threshold': 0.9,
                'line_curve_ratio': (0.7, 1.0),
                'complexity_range': (0.7, 0.95)
            }
        }
        
        # Geometric feature thresholds
        self.geometric_thresholds = {
            'min_contour_area': 50,
            'max_contour_area': 10000,
            'aspect_ratio_threshold': 0.1,
            'solidity_threshold': 0.3,
            'extent_threshold': 0.2
        }
        
        # Topological analysis parameters
        self.topology_params = {
            'skeleton_threshold': 0.1,
            'branch_length_threshold': 10,
            'junction_distance_threshold': 5,
            'loop_area_threshold': 100
        }
    
    def analyze_topology(self, image: np.ndarray, skeleton: Optional[np.ndarray] = None) -> Dict[str, Union[int, float, List]]:
        """
        Analyze topological properties of the pattern including connectivity, loops, and junctions.
        
        Args:
            image: Binary input image
            skeleton: Optional pre-computed skeleton
            
        Returns:
            Dictionary containing topological features
        """
        try:
            # Ensure binary image
            if len(image.shape) == 3:
                binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                binary = image.copy()
            
            _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
            
            # Compute skeleton if not provided
            if skeleton is None:
                skeleton = self._compute_skeleton(binary)
            
            topology_features = {}
            
            # Basic connectivity analysis
            num_components, labels = cv2.connectedComponents(binary)
            topology_features['num_components'] = num_components - 1  # Subtract background
            
            # Euler number (topological invariant)
            euler_number = self._calculate_euler_number(binary)
            topology_features['euler_number'] = euler_number
            
            # Junction analysis
            junctions = self._detect_junctions(skeleton)
            topology_features['num_junctions'] = len(junctions)
            topology_features['junction_coordinates'] = junctions
            
            # Branch analysis
            branches = self._analyze_branches(skeleton, junctions)
            topology_features['num_branches'] = len(branches)
            topology_features['avg_branch_length'] = np.mean([b['length'] for b in branches]) if branches else 0
            topology_features['branch_length_std'] = np.std([b['length'] for b in branches]) if branches else 0
            
            # Loop detection and analysis
            loops = self._detect_loops(skeleton)
            topology_features['num_loops'] = len(loops)
            topology_features['avg_loop_area'] = np.mean([l['area'] for l in loops]) if loops else 0
            
            # Graph-based analysis
            graph_features = self._analyze_pattern_graph(skeleton, junctions)
            topology_features.update(graph_features)
            
            # Structural complexity
            topology_features['structural_complexity'] = self._calculate_structural_complexity(
                topology_features
            )
            
            logger.info(f"Topology analysis completed: {topology_features['num_components']} components, "
                       f"{topology_features['num_junctions']} junctions, {topology_features['num_loops']} loops")
            
        except Exception as e:
            logger.error(f"Error in topology analysis: {e}")
            topology_features = self._get_default_topology_features()
        
        return topology_features
    
    def measure_complexity(self, image: np.ndarray, 
                         features: Optional[Dict] = None) -> Dict[str, float]:
        """
        Measure pattern complexity using multiple metrics.
        
        Args:
            image: Input image (binary or grayscale)
            features: Optional pre-computed features
            
        Returns:
            Dictionary containing complexity scores
        """
        try:
            complexity_scores = {}
            
            # Ensure grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Geometric complexity
            complexity_scores['geometric'] = self._calculate_geometric_complexity(gray)
            
            # Fractal dimension
            complexity_scores['fractal'] = self._calculate_fractal_dimension(gray)
            
            # Information-theoretic complexity
            complexity_scores['information'] = self._calculate_information_complexity(gray)
            
            # Structural complexity (from topology if available)
            if features and 'structural_complexity' in features:
                complexity_scores['structural'] = features['structural_complexity']
            else:
                complexity_scores['structural'] = self._estimate_structural_complexity(gray)
            
            # Symmetry-based complexity
            complexity_scores['symmetry'] = self._calculate_symmetry_complexity(gray)
            
            # Visual complexity (edge density and variation)
            complexity_scores['visual'] = self._calculate_visual_complexity(gray)
            
            # Combined complexity score
            complexity_scores['overall'] = self._combine_complexity_scores(complexity_scores)
            
            logger.info(f"Complexity analysis completed. Overall score: {complexity_scores['overall']:.3f}")
            
        except Exception as e:
            logger.error(f"Error measuring complexity: {e}")
            complexity_scores = {
                'geometric': 0.5,
                'fractal': 0.5,
                'information': 0.5,
                'structural': 0.5,
                'symmetry': 0.5,
                'visual': 0.5,
                'overall': 0.5
            }
        
        return complexity_scores
    
    def detect_pattern_type(self, image: np.ndarray, 
                          features: Optional[Dict] = None) -> Dict[str, Union[str, float, Dict]]:
        """
        Detect and classify the type of Kolam pattern.
        
        Args:
            image: Input image
            features: Optional pre-computed features
            
        Returns:
            Dictionary containing pattern type classification results
        """
        try:
            # Compute features if not provided
            if features is None:
                features = self._extract_comprehensive_features(image)
            
            classification_results = {}
            pattern_scores = {}
            
            # Calculate similarity scores for each pattern type
            for pattern_type, signature in self.pattern_signatures.items():
                score = self._calculate_pattern_similarity(features, signature)
                pattern_scores[pattern_type] = score
            
            # Determine best match
            best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
            
            classification_results['pattern_type'] = best_pattern[0]
            classification_results['confidence'] = best_pattern[1]
            classification_results['all_scores'] = pattern_scores
            
            # Additional pattern characteristics
            classification_results['characteristics'] = self._analyze_pattern_characteristics(features)
            
            # Cultural authenticity score
            classification_results['authenticity_score'] = self._calculate_authenticity_score(features)
            
            logger.info(f"Pattern type detected: {classification_results['pattern_type']} "
                       f"(confidence: {classification_results['confidence']:.3f})")
            
        except Exception as e:
            logger.error(f"Error in pattern type detection: {e}")
            classification_results = {
                'pattern_type': 'unknown',
                'confidence': 0.0,
                'all_scores': {},
                'characteristics': {},
                'authenticity_score': 0.0
            }
        
        return classification_results
    
    def calculate_metrics(self, image: np.ndarray) -> Dict[str, Union[float, int, Dict]]:
        """
        Calculate comprehensive pattern metrics including cultural and mathematical properties.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing all pattern metrics
        """
        try:
            metrics = {}
            
            # Basic image properties
            metrics['image_properties'] = self._calculate_basic_properties(image)
            
            # Geometric features
            metrics['geometric_features'] = self._extract_geometric_features(image)
            
            # Topological analysis
            metrics['topological_features'] = self.analyze_topology(image)
            
            # Complexity measures
            metrics['complexity'] = self.measure_complexity(image, metrics['topological_features'])
            
            # Pattern classification
            combined_features = {**metrics['geometric_features'], **metrics['topological_features']}
            metrics['classification'] = self.detect_pattern_type(image, combined_features)
            
            # Cultural metrics
            metrics['cultural_metrics'] = self._calculate_cultural_metrics(image, combined_features)
            
            # Mathematical properties
            metrics['mathematical_properties'] = self._calculate_mathematical_properties(image)
            
            # Quality assessment
            metrics['quality_assessment'] = self._assess_pattern_quality(image, metrics)
            
            logger.info("Comprehensive pattern metrics calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            metrics = self._get_default_metrics()
        
        return metrics
    
    # Private helper methods
    def _compute_skeleton(self, binary_image: np.ndarray) -> np.ndarray:
        """Compute skeleton using morphological operations."""
        skeleton = np.zeros(binary_image.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            eroded = cv2.erode(binary_image, element)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            subset = cv2.subtract(eroded, opened)
            skeleton = cv2.bitwise_or(skeleton, subset)
            binary_image = eroded.copy()
            
            if cv2.countNonZero(binary_image) == 0:
                break
        
        return skeleton
    
    def _calculate_euler_number(self, binary_image: np.ndarray) -> int:
        """Calculate Euler number (topological invariant)."""
        # Count connected components
        num_components, _ = cv2.connectedComponents(binary_image)
        components = num_components - 1  # Subtract background
        
        # Count holes using contour hierarchy
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        holes = 0
        if hierarchy is not None:
            # Count internal contours (holes)
            for i in range(len(hierarchy[0])):
                if hierarchy[0][i][3] != -1:  # Has a parent (internal contour)
                    holes += 1
        
        # Euler number = Components - Holes
        euler_number = components - holes
        
        return euler_number
    
    def _detect_junctions(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Detect junction points in skeleton."""
        # Create kernel to count neighbors
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        # Count neighbors for each point
        neighbor_count = cv2.filter2D(skeleton, -1, kernel)
        
        # Junctions have more than 2 neighbors
        junctions_mask = (skeleton > 0) & (neighbor_count > 2 * 255)
        
        # Get junction coordinates
        junction_coords = np.where(junctions_mask)
        junctions = list(zip(junction_coords[1], junction_coords[0]))  # (x, y) format
        
        return junctions
    
    def _analyze_branches(self, skeleton: np.ndarray, junctions: List[Tuple[int, int]]) -> List[Dict]:
        """Analyze branches in the skeleton."""
        branches = []
        
        # Create junction mask
        junction_mask = np.zeros_like(skeleton)
        for x, y in junctions:
            junction_mask[y, x] = 255
        
        # Remove junctions to isolate branches
        skeleton_no_junctions = cv2.subtract(skeleton, junction_mask)
        
        # Find connected components (branches)
        num_labels, labels = cv2.connectedComponents(skeleton_no_junctions)
        
        for label in range(1, num_labels):
            branch_mask = (labels == label).astype(np.uint8) * 255
            
            # Calculate branch properties
            branch_pixels = np.where(branch_mask > 0)
            if len(branch_pixels[0]) > 0:
                # Branch length (approximate)
                length = len(branch_pixels[0])
                
                # Branch endpoints
                endpoints = self._find_endpoints(branch_mask)
                
                # Branch curvature (simplified)
                curvature = self._calculate_branch_curvature(branch_pixels)
                
                branches.append({
                    'length': length,
                    'endpoints': endpoints,
                    'curvature': curvature,
                    'pixels': list(zip(branch_pixels[1], branch_pixels[0]))
                })
        
        return branches
    
    def _detect_loops(self, skeleton: np.ndarray) -> List[Dict]:
        """Detect loops in the pattern."""
        loops = []
        
        # Find contours in skeleton
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Check if contour is closed (forms a loop)
            area = cv2.contourArea(contour)
            if area > self.topology_params['loop_area_threshold']:
                perimeter = cv2.arcLength(contour, True)
                
                # Calculate roundness (4π*area/perimeter²)
                if perimeter > 0:
                    roundness = (4 * np.pi * area) / (perimeter ** 2)
                else:
                    roundness = 0
                
                loops.append({
                    'area': area,
                    'perimeter': perimeter,
                    'roundness': roundness,
                    'contour': contour
                })
        
        return loops
    
    def _analyze_pattern_graph(self, skeleton: np.ndarray, junctions: List[Tuple[int, int]]) -> Dict:
        """Analyze pattern as a graph structure."""
        graph_features = {}
        
        try:
            # Create graph
            G = nx.Graph()
            
            # Add junction nodes
            for i, junction in enumerate(junctions):
                G.add_node(i, pos=junction)
            
            # Find connections between junctions
            # This is a simplified approach - in practice, you'd trace paths
            for i, j1 in enumerate(junctions):
                for j, j2 in enumerate(junctions[i+1:], i+1):
                    # Check if junctions are connected via skeleton path
                    if self._are_junctions_connected(skeleton, j1, j2):
                        distance = np.sqrt((j1[0] - j2[0])**2 + (j1[1] - j2[1])**2)
                        G.add_edge(i, j, weight=distance)
            
            # Graph properties
            graph_features['num_nodes'] = G.number_of_nodes()
            graph_features['num_edges'] = G.number_of_edges()
            
            if G.number_of_nodes() > 0:
                graph_features['density'] = nx.density(G)
                graph_features['avg_clustering'] = nx.average_clustering(G)
                
                # Degree statistics
                degrees = [G.degree(n) for n in G.nodes()]
                graph_features['avg_degree'] = np.mean(degrees) if degrees else 0
                graph_features['max_degree'] = max(degrees) if degrees else 0
                
                # Path lengths
                if nx.is_connected(G):
                    graph_features['diameter'] = nx.diameter(G)
                    graph_features['avg_path_length'] = nx.average_shortest_path_length(G)
                else:
                    graph_features['diameter'] = 0
                    graph_features['avg_path_length'] = 0
                    
            else:
                graph_features.update({
                    'density': 0, 'avg_clustering': 0, 'avg_degree': 0,
                    'max_degree': 0, 'diameter': 0, 'avg_path_length': 0
                })
                
        except Exception as e:
            logger.warning(f"Error in graph analysis: {e}")
            graph_features = {
                'num_nodes': len(junctions), 'num_edges': 0, 'density': 0,
                'avg_clustering': 0, 'avg_degree': 0, 'max_degree': 0,
                'diameter': 0, 'avg_path_length': 0
            }
        
        return graph_features
    
    def _calculate_structural_complexity(self, topology_features: Dict) -> float:
        """Calculate structural complexity from topological features."""
        # Normalize components
        components = min(topology_features.get('num_components', 1), 10) / 10.0
        
        # Normalize junctions
        junctions = min(topology_features.get('num_junctions', 0), 20) / 20.0
        
        # Normalize branches
        branches = min(topology_features.get('num_branches', 0), 30) / 30.0
        
        # Normalize loops
        loops = min(topology_features.get('num_loops', 0), 10) / 10.0
        
        # Graph complexity
        density = topology_features.get('density', 0)
        clustering = topology_features.get('avg_clustering', 0)
        
        # Combine features
        structural_complexity = (
            0.2 * components +
            0.25 * junctions +
            0.25 * branches +
            0.15 * loops +
            0.1 * density +
            0.05 * clustering
        )
        
        return min(structural_complexity, 1.0)
    
    def _calculate_geometric_complexity(self, image: np.ndarray) -> float:
        """Calculate geometric complexity based on shapes and contours."""
        # Find contours
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        complexity_factors = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.geometric_thresholds['min_contour_area']:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            
            # Shape irregularity
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                irregularity = 1.0 - circularity
            else:
                irregularity = 0
            
            # Convexity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                convexity_defect = 1.0 - (area / hull_area)
            else:
                convexity_defect = 0
            
            # Aspect ratio variation
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if height > 0:
                aspect_ratio = width / height
                aspect_complexity = min(aspect_ratio, 1.0 / aspect_ratio)
            else:
                aspect_complexity = 0
            
            contour_complexity = (irregularity + convexity_defect + (1 - aspect_complexity)) / 3.0
            complexity_factors.append(contour_complexity)
        
        return np.mean(complexity_factors) if complexity_factors else 0.0
    
    def _calculate_fractal_dimension(self, image: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method."""
        # Ensure binary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Box counting
        def box_count(image, box_size):
            h, w = image.shape
            count = 0
            
            for i in range(0, h, box_size):
                for j in range(0, w, box_size):
                    box = image[i:i+box_size, j:j+box_size]
                    if np.any(box):
                        count += 1
            
            return count
        
        # Different box sizes
        sizes = [2, 4, 8, 16, 32]
        counts = []
        
        for size in sizes:
            if size < min(binary.shape):
                count = box_count(binary, size)
                counts.append(count)
            else:
                break
        
        if len(counts) < 2:
            return 1.0  # Default fractal dimension
        
        # Linear regression in log-log plot
        log_sizes = np.log(sizes[:len(counts)])
        log_counts = np.log(counts)
        
        # Fit line: log(count) = -D * log(size) + constant
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        fractal_dimension = -coeffs[0]
        
        # Normalize to 0-1 range (typical range for 2D patterns is 1-2)
        normalized_fd = max(0, min(1, (fractal_dimension - 1.0)))
        
        return normalized_fd
    
    def _calculate_information_complexity(self, image: np.ndarray) -> float:
        """Calculate information-theoretic complexity (entropy)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate histogram
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        
        # Normalize to get probability distribution
        hist = hist.astype(float)
        hist = hist / np.sum(hist)
        
        # Remove zero probabilities
        hist = hist[hist > 0]
        
        # Calculate entropy
        if len(hist) > 1:
            entropy_value = -np.sum(hist * np.log2(hist))
            # Normalize (max entropy for 256 bins is log2(256) = 8)
            normalized_entropy = entropy_value / 8.0
        else:
            normalized_entropy = 0.0
        
        return normalized_entropy
    
    def _estimate_structural_complexity(self, image: np.ndarray) -> float:
        """Estimate structural complexity without full topology analysis."""
        # Quick structural features
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Connected components
        num_components, _ = cv2.connectedComponents(binary)
        component_complexity = min((num_components - 1) / 10.0, 1.0)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Contour complexity
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_complexity = min(len(contours) / 20.0, 1.0)
        
        structural_estimate = (component_complexity + edge_density + contour_complexity) / 3.0
        
        return structural_estimate
    
    def _calculate_symmetry_complexity(self, image: np.ndarray) -> float:
        """Calculate complexity based on symmetry patterns."""
        # This would ideally use the ImageProcessor's symmetry analysis
        # For now, implement a basic version
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        height, width = gray.shape
        
        # Vertical symmetry
        left_half = gray[:, :width//2]
        right_half = cv2.flip(gray[:, width//2:], 1)
        
        if left_half.shape != right_half.shape:
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
        
        vertical_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
        vertical_symmetry = max(0, 1 - vertical_diff / 255.0)
        
        # Horizontal symmetry
        top_half = gray[:height//2, :]
        bottom_half = cv2.flip(gray[height//2:, :], 0)
        
        if top_half.shape != bottom_half.shape:
            min_height = min(top_half.shape[0], bottom_half.shape[0])
            top_half = top_half[:min_height, :]
            bottom_half = bottom_half[:min_height, :]
        
        horizontal_diff = np.mean(np.abs(top_half.astype(float) - bottom_half.astype(float)))
        horizontal_symmetry = max(0, 1 - horizontal_diff / 255.0)
        
        # Higher symmetry means lower complexity in this context
        avg_symmetry = (vertical_symmetry + horizontal_symmetry) / 2.0
        symmetry_complexity = 1.0 - avg_symmetry
        
        return symmetry_complexity
    
    def _calculate_visual_complexity(self, image: np.ndarray) -> float:
        """Calculate visual complexity based on edge density and variations."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
        gradient_complexity = np.mean(gradient_magnitude)
        
        # Texture variation
        texture_variance = np.var(gray.astype(float)) / (255.0 ** 2)
        
        visual_complexity = (edge_density + gradient_complexity + texture_variance) / 3.0
        
        return min(visual_complexity, 1.0)
    
    def _combine_complexity_scores(self, scores: Dict[str, float]) -> float:
        """Combine individual complexity scores into overall score."""
        # Use predefined weights
        overall_score = 0.0
        total_weight = 0.0
        
        score_mapping = {
            'geometric': 'geometric',
            'fractal': 'fractal',
            'information': 'topological',
            'structural': 'structural',
            'symmetry': 'symmetry',
            'visual': 'density'
        }
        
        for score_name, weight_name in score_mapping.items():
            if score_name in scores and weight_name in self.complexity_weights:
                weight = self.complexity_weights[weight_name]
                overall_score += scores[score_name] * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_score = overall_score / total_weight
        
        return min(overall_score, 1.0)
    
    def _extract_comprehensive_features(self, image: np.ndarray) -> Dict:
        """Extract comprehensive features for pattern analysis."""
        features = {}
        
        # Basic geometric features
        features.update(self._extract_geometric_features(image))
        
        # Topological features
        topology = self.analyze_topology(image)
        features.update(topology)
        
        # Complexity measures
        complexity = self.measure_complexity(image)
        features.update(complexity)
        
        return features
    
    def _extract_geometric_features(self, image: np.ndarray) -> Dict:
        """Extract geometric features from the image."""
        features = {}
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._get_default_geometric_features()
        
        # Contour statistics
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
        perimeters = [cv2.arcLength(c, True) for c in contours if cv2.contourArea(c) > 10]
        
        features['num_contours'] = len(areas)
        features['avg_area'] = np.mean(areas) if areas else 0
        features['area_std'] = np.std(areas) if areas else 0
        features['avg_perimeter'] = np.mean(perimeters) if perimeters else 0
        features['perimeter_std'] = np.std(perimeters) if perimeters else 0
        
        # Shape analysis
        if areas:
            # Circularity
            circularities = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 10:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = (4 * np.pi * area) / (perimeter ** 2)
                        circularities.append(circularity)
            
            features['avg_circularity'] = np.mean(circularities) if circularities else 0
            features['circularity_std'] = np.std(circularities) if circularities else 0
        
        # Density features
        total_area = np.sum(binary > 0)
        image_area = binary.shape[0] * binary.shape[1]
        features['density'] = total_area / image_area if image_area > 0 else 0
        
        # Dot detection (circular patterns)
        circles = self._detect_dots(gray)
        features['num_dots'] = len(circles)
        features['dot_density'] = len(circles) / image_area if image_area > 0 else 0
        
        # Line vs curve analysis
        features['line_curve_ratio'] = self._analyze_line_curve_ratio(binary)
        
        return features
    
    def _detect_dots(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect circular dots in the pattern."""
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=3, maxRadius=30
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return [(int(x), int(y), int(r)) for x, y, r in circles]
        else:
            return []
    
    def _analyze_line_curve_ratio(self, binary_image: np.ndarray) -> float:
        """Analyze the ratio of straight lines to curves."""
        # Detect lines
        edges = cv2.Canny(binary_image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                               minLineLength=20, maxLineGap=10)
        
        line_length = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                line_length += length
        
        # Total edge length
        total_edge_length = np.sum(edges > 0)
        
        if total_edge_length > 0:
            line_ratio = line_length / total_edge_length
        else:
            line_ratio = 0
        
        return min(line_ratio, 1.0)
    
    def _calculate_pattern_similarity(self, features: Dict, signature: Dict) -> float:
        """Calculate similarity between pattern features and type signature."""
        score = 0.0
        num_matches = 0
        
        # Dot density check
        if 'dot_density' in features:
            dot_density = features['dot_density']
            dot_range = signature['dot_density_range']
            if dot_range[0] <= dot_density <= dot_range[1]:
                score += 1.0
            else:
                # Partial score based on distance to range
                if dot_density < dot_range[0]:
                    distance = dot_range[0] - dot_density
                else:
                    distance = dot_density - dot_range[1]
                score += max(0, 1 - distance / 0.1)  # Penalty decreases with distance
            num_matches += 1
        
        # Symmetry check
        if 'overall' in features and 'symmetry' in features:
            symmetry_score = features.get('overall', 0)  # From symmetry analysis
            threshold = signature['symmetry_threshold']
            if symmetry_score >= threshold:
                score += 1.0
            else:
                score += symmetry_score / threshold
            num_matches += 1
        
        # Line-curve ratio check
        if 'line_curve_ratio' in features:
            line_curve = features['line_curve_ratio']
            ratio_range = signature['line_curve_ratio']
            if ratio_range[0] <= line_curve <= ratio_range[1]:
                score += 1.0
            else:
                if line_curve < ratio_range[0]:
                    distance = ratio_range[0] - line_curve
                else:
                    distance = line_curve - ratio_range[1]
                score += max(0, 1 - distance / 0.3)
            num_matches += 1
        
        # Complexity check
        if 'overall' in features:
            complexity = features['overall']
            complexity_range = signature['complexity_range']
            if complexity_range[0] <= complexity <= complexity_range[1]:
                score += 1.0
            else:
                if complexity < complexity_range[0]:
                    distance = complexity_range[0] - complexity
                else:
                    distance = complexity - complexity_range[1]
                score += max(0, 1 - distance / 0.2)
            num_matches += 1
        
        return score / num_matches if num_matches > 0 else 0.0
    
    def _analyze_pattern_characteristics(self, features: Dict) -> Dict:
        """Analyze specific pattern characteristics."""
        characteristics = {}
        
        # Structural characteristics
        characteristics['is_grid_based'] = features.get('num_dots', 0) > 10
        characteristics['has_central_symmetry'] = features.get('rotational_180', 0) > 0.7
        characteristics['is_geometric'] = features.get('avg_circularity', 0) > 0.6
        characteristics['is_organic'] = features.get('avg_circularity', 0) < 0.4
        
        # Complexity characteristics
        complexity = features.get('overall', 0.5)
        if complexity < 0.3:
            characteristics['complexity_level'] = 'simple'
        elif complexity < 0.7:
            characteristics['complexity_level'] = 'moderate'
        else:
            characteristics['complexity_level'] = 'complex'
        
        # Topological characteristics
        num_components = features.get('num_components', 1)
        num_loops = features.get('num_loops', 0)
        
        characteristics['is_connected'] = num_components <= 3
        characteristics['has_loops'] = num_loops > 0
        characteristics['loop_dominant'] = num_loops > num_components
        
        return characteristics
    
    def _calculate_authenticity_score(self, features: Dict) -> float:
        """Calculate cultural authenticity score based on traditional Kolam principles."""
        authenticity_score = 0.0
        criteria_count = 0
        
        # Symmetry criterion (traditional Kolams are highly symmetric)
        if 'overall' in features:
            symmetry = features.get('overall', 0)
            authenticity_score += min(symmetry * 1.5, 1.0)  # Bonus for high symmetry
            criteria_count += 1
        
        # Grid-based criterion (many traditional Kolams use dot grids)
        if features.get('num_dots', 0) > 0:
            dot_density = features.get('dot_density', 0)
            if 0.05 <= dot_density <= 0.25:  # Optimal dot density range
                authenticity_score += 1.0
            else:
                authenticity_score += 0.5
            criteria_count += 1
        
        # Continuous line criterion (Sikku Kolam principle)
        num_components = features.get('num_components', 1)
        if num_components == 1:  # Single continuous line
            authenticity_score += 1.0
        elif num_components <= 3:
            authenticity_score += 0.7
        else:
            authenticity_score += 0.3
        criteria_count += 1
        
        # Mathematical proportions (golden ratio, etc.)
        # This is simplified - real implementation would check geometric ratios
        geometric_score = features.get('geometric', 0.5)
        authenticity_score += geometric_score
        criteria_count += 1
        
        # Loop closure criterion (traditional Kolams often have closed loops)
        num_loops = features.get('num_loops', 0)
        if num_loops > 0:
            authenticity_score += 1.0
        else:
            authenticity_score += 0.3
        criteria_count += 1
        
        return authenticity_score / criteria_count if criteria_count > 0 else 0.0
    
    def _calculate_basic_properties(self, image: np.ndarray) -> Dict:
        """Calculate basic image properties."""
        properties = {}
        
        if len(image.shape) == 3:
            properties['channels'] = image.shape[2]
            properties['is_color'] = True
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            properties['channels'] = 1
            properties['is_color'] = False
            gray = image
        
        properties['height'], properties['width'] = gray.shape
        properties['total_pixels'] = gray.shape[0] * gray.shape[1]
        properties['aspect_ratio'] = properties['width'] / properties['height']
        
        # Pixel intensity statistics
        properties['mean_intensity'] = np.mean(gray)
        properties['std_intensity'] = np.std(gray)
        properties['min_intensity'] = np.min(gray)
        properties['max_intensity'] = np.max(gray)
        
        return properties
    
    def _calculate_cultural_metrics(self, image: np.ndarray, features: Dict) -> Dict:
        """Calculate metrics specific to cultural authenticity and traditional principles."""
        cultural_metrics = {}
        
        # Traditional Kolam principles
        cultural_metrics['mathematical_harmony'] = self._assess_mathematical_harmony(features)
        cultural_metrics['spiritual_symbolism'] = self._assess_spiritual_symbolism(features)
        cultural_metrics['seasonal_appropriateness'] = self._assess_seasonal_appropriateness(features)
        cultural_metrics['regional_style'] = self._identify_regional_style(features)
        cultural_metrics['difficulty_level'] = self._assess_difficulty_level(features)
        
        return cultural_metrics
    
    def _calculate_mathematical_properties(self, image: np.ndarray) -> Dict:
        """Calculate mathematical properties of the pattern."""
        properties = {}
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Fourier analysis
        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        
        properties['frequency_peak'] = np.max(fft_magnitude[1:, 1:])  # Exclude DC component
        properties['frequency_mean'] = np.mean(fft_magnitude[1:, 1:])
        
        # Moments (Hu moments for shape description)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(binary)
        
        if moments['m00'] != 0:
            # Centroids
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            properties['centroid_x'] = cx
            properties['centroid_y'] = cy
            
            # Hu moments
            hu_moments = cv2.HuMoments(moments)
            for i, hu in enumerate(hu_moments.flatten()):
                properties[f'hu_moment_{i+1}'] = -np.sign(hu) * np.log10(abs(hu)) if hu != 0 else 0
        
        return properties
    
    def _assess_pattern_quality(self, image: np.ndarray, metrics: Dict) -> Dict:
        """Assess the quality of the pattern."""
        quality = {}
        
        # Drawing quality (smoothness, clarity)
        quality['line_smoothness'] = self._assess_line_smoothness(image)
        quality['clarity'] = self._assess_pattern_clarity(image)
        quality['completeness'] = self._assess_pattern_completeness(metrics)
        
        # Overall quality score
        quality_scores = [quality['line_smoothness'], quality['clarity'], quality['completeness']]
        quality['overall_quality'] = np.mean(quality_scores)
        
        return quality
    
    # Additional helper methods for cultural and mathematical analysis
    def _assess_mathematical_harmony(self, features: Dict) -> float:
        """Assess mathematical harmony in the pattern."""
        # Check for golden ratio, symmetrical proportions, etc.
        # Simplified implementation
        symmetry_score = features.get('overall', 0.5)
        geometric_score = features.get('geometric', 0.5)
        
        harmony_score = (symmetry_score + geometric_score) / 2.0
        return harmony_score
    
    def _assess_spiritual_symbolism(self, features: Dict) -> float:
        """Assess spiritual and symbolic elements."""
        # Check for traditional symbolic patterns
        # This would require a database of symbolic patterns
        
        # Placeholder: based on complexity and symmetry
        complexity = features.get('overall', 0.5)
        symmetry = features.get('overall', 0.5)  # Would be symmetry score
        
        symbolism_score = (complexity * 0.3 + symmetry * 0.7)
        return symbolism_score
    
    def _assess_seasonal_appropriateness(self, features: Dict) -> float:
        """Assess appropriateness for different seasons/festivals."""
        # This would require seasonal pattern classification
        # Placeholder implementation
        complexity = features.get('overall', 0.5)
        
        # Simple festivals typically use simpler patterns
        if complexity < 0.4:
            return 0.8  # Good for daily use
        elif complexity < 0.7:
            return 0.9  # Good for weekly festivals
        else:
            return 1.0  # Good for major festivals
    
    def _identify_regional_style(self, features: Dict) -> str:
        """Identify regional style based on pattern characteristics."""
        # This would require regional pattern classification
        
        dot_density = features.get('dot_density', 0)
        complexity = features.get('overall', 0.5)
        
        if dot_density > 0.15 and complexity < 0.6:
            return 'Tamil Nadu - Simple'
        elif dot_density < 0.1 and complexity > 0.7:
            return 'Tamil Nadu - Advanced'
        else:
            return 'General South Indian'
    
    def _assess_difficulty_level(self, features: Dict) -> str:
        """Assess difficulty level for drawing."""
        complexity = features.get('overall', 0.5)
        num_components = features.get('num_components', 1)
        num_junctions = features.get('num_junctions', 0)
        
        difficulty_score = (complexity + 
                          min(num_components / 10.0, 0.3) + 
                          min(num_junctions / 20.0, 0.3))
        
        if difficulty_score < 0.3:
            return 'Beginner'
        elif difficulty_score < 0.6:
            return 'Intermediate'
        elif difficulty_score < 0.8:
            return 'Advanced'
        else:
            return 'Expert'
    
    def _assess_line_smoothness(self, image: np.ndarray) -> float:
        """Assess smoothness of lines in the pattern."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate second derivative (curvature)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        smoothness = 1.0 - (np.std(laplacian) / 255.0)
        
        return max(0, min(1, smoothness))
    
    def _assess_pattern_clarity(self, image: np.ndarray) -> float:
        """Assess clarity and contrast of the pattern."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Edge strength
        edges = cv2.Canny(gray, 50, 150)
        edge_strength = np.mean(edges) / 255.0
        
        # Contrast
        contrast = np.std(gray) / 128.0
        
        clarity = (edge_strength + contrast) / 2.0
        return min(clarity, 1.0)
    
    def _assess_pattern_completeness(self, metrics: Dict) -> float:
        """Assess completeness of the pattern."""
        # Check for broken lines, incomplete shapes
        
        geometric_features = metrics.get('geometric_features', {})
        topological_features = metrics.get('topological_features', {})
        
        # Simple heuristic based on connectivity
        num_components = topological_features.get('num_components', 1)
        expected_components = 1  # Ideally one connected component
        
        completeness = max(0, 1 - abs(num_components - expected_components) / 10.0)
        
        return completeness
    
    # Helper methods for endpoints and curvature
    def _find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find endpoints in a skeleton."""
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        neighbor_count = cv2.filter2D(skeleton, -1, kernel)
        endpoints_mask = (skeleton > 0) & (neighbor_count == 255)  # Only 1 neighbor
        
        endpoint_coords = np.where(endpoints_mask)
        return list(zip(endpoint_coords[1], endpoint_coords[0]))
    
    def _calculate_branch_curvature(self, branch_pixels: Tuple) -> float:
        """Calculate curvature of a branch."""
        if len(branch_pixels[0]) < 3:
            return 0.0
        
        # Simple curvature estimation using angle changes
        y_coords, x_coords = branch_pixels
        points = list(zip(x_coords, y_coords))
        
        if len(points) < 3:
            return 0.0
        
        angles = []
        for i in range(1, len(points) - 1):
            p1, p2, p3 = points[i-1], points[i], points[i+1]
            
            # Vector angles
            v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Angle between vectors
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        # Average curvature
        if angles:
            return np.mean(angles) / np.pi  # Normalize to 0-1
        else:
            return 0.0
    
    def _are_junctions_connected(self, skeleton: np.ndarray, j1: Tuple[int, int], j2: Tuple[int, int]) -> bool:
        """Check if two junctions are connected via skeleton path."""
        # Simple flood fill to check connectivity
        # This is a simplified implementation
        
        temp_skeleton = skeleton.copy()
        
        # Start flood fill from j1
        stack = [j1]
        visited = set()
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == j2:
                return True
            
            x, y = current
            
            # Check 8-connected neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    
                    if (0 <= nx < temp_skeleton.shape[1] and 
                        0 <= ny < temp_skeleton.shape[0] and
                        temp_skeleton[ny, nx] > 0 and
                        (nx, ny) not in visited):
                        stack.append((nx, ny))
        
        return False
    
    # Default return methods
    def _get_default_topology_features(self) -> Dict:
        """Return default topology features."""
        return {
            'num_components': 1,
            'euler_number': 1,
            'num_junctions': 0,
            'junction_coordinates': [],
            'num_branches': 0,
            'avg_branch_length': 0,
            'branch_length_std': 0,
            'num_loops': 0,
            'avg_loop_area': 0,
            'structural_complexity': 0.5,
            'num_nodes': 0,
            'num_edges': 0,
            'density': 0,
            'avg_clustering': 0,
            'avg_degree': 0,
            'max_degree': 0,
            'diameter': 0,
            'avg_path_length': 0
        }
    
    def _get_default_geometric_features(self) -> Dict:
        """Return default geometric features."""
        return {
            'num_contours': 0,
            'avg_area': 0,
            'area_std': 0,
            'avg_perimeter': 0,
            'perimeter_std': 0,
            'avg_circularity': 0,
            'circularity_std': 0,
            'density': 0,
            'num_dots': 0,
            'dot_density': 0,
            'line_curve_ratio': 0
        }
    
    def _get_default_metrics(self) -> Dict:
        """Return default comprehensive metrics."""
        return {
            'image_properties': {'height': 0, 'width': 0, 'channels': 1, 'is_color': False},
            'geometric_features': self._get_default_geometric_features(),
            'topological_features': self._get_default_topology_features(),
            'complexity': {'overall': 0.5},
            'classification': {'pattern_type': 'unknown', 'confidence': 0.0},
            'cultural_metrics': {},
            'mathematical_properties': {},
            'quality_assessment': {'overall_quality': 0.0}
        }

# Testing and example usage
def test_pattern_analyzer():
    """Test the PatternAnalyzer class with sample patterns."""
    analyzer = PatternAnalyzer()
    
    # Create test pattern
    test_image = np.zeros((200, 200), dtype=np.uint8)
    
    # Draw a simple Kolam-like pattern
    center = (100, 100)
    
    # Draw dots in grid
    for i in range(5):
        for j in range(5):
            x, y = 50 + i * 25, 50 + j * 25
            cv2.circle(test_image, (x, y), 3, 255, -1)
    
    # Draw connecting lines
    cv2.line(test_image, (50, 50), (150, 50), 255, 2)
    cv2.line(test_image, (150, 50), (150, 150), 255, 2)
    cv2.line(test_image, (150, 150), (50, 150), 255, 2)
    cv2.line(test_image, (50, 150), (50, 50), 255, 2)
    
    print("Testing PatternAnalyzer...")
    
    # Test topology analysis
    topology = analyzer.analyze_topology(test_image)
    print(f"Topology: {topology['num_components']} components, {topology['num_junctions']} junctions")
    
    # Test complexity measurement
    complexity = analyzer.measure_complexity(test_image)
    print(f"Complexity: {complexity['overall']:.3f}")
    
    # Test pattern type detection
    classification = analyzer.detect_pattern_type(test_image)
    print(f"Pattern type: {classification['pattern_type']} (confidence: {classification['confidence']:.3f})")
    
    # Test comprehensive metrics
    metrics = analyzer.calculate_metrics(test_image)
    print(f"Quality assessment: {metrics['quality_assessment']['overall_quality']:.3f}")
    
    print("PatternAnalyzer test completed!")

if __name__ == "__main__":
    test_pattern_analyzer()