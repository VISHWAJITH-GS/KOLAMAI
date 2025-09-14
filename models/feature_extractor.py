import cv2
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import measure, morphology
from skimage.feature import graycomatrix, graycoprops
import json

class FeatureExtractor:
    """
    Computer vision feature extraction module for Kolam pattern analysis.
    Extracts geometric features, symmetry measures, and cultural pattern signatures.
    """
    
    def __init__(self):
        # Feature extraction parameters
        self.symmetry_threshold = 0.85
        self.dot_detection_params = {
            'min_radius': 3,
            'max_radius': 15,
            'param1': 50,
            'param2': 30,
            'min_distance': 10
        }
        
        # Morphological kernel sizes
        self.kernel_sizes = {
            'small': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            'medium': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            'large': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        }
        
        # Complexity analysis parameters
        self.complexity_weights = {
            'contour_count': 0.2,
            'area_variation': 0.15,
            'curvature_variation': 0.2,
            'intersection_points': 0.25,
            'symmetry_deviation': 0.2
        }
        
        # Pattern signature parameters
        self.signature_radii = [10, 20, 30, 40, 50]
        self.angular_bins = 36  # 10-degree bins
        
    def extract_all_features(self, image: np.ndarray) -> Dict:
        """
        Extract comprehensive feature set from Kolam image.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            Dictionary containing all extracted features
        """
        # Preprocess image
        processed_img = self._preprocess_image(image)
        
        # Extract different feature categories
        geometric_features = self.extract_geometric_features(processed_img)
        symmetry_features = self.calculate_symmetry(processed_img)
        dot_features = self.detect_dots(processed_img)
        curve_features = self.analyze_curves(processed_img)
        topological_features = self._extract_topological_features(processed_img)
        texture_features = self._extract_texture_features(processed_img)
        cultural_signature = self._extract_cultural_signature(processed_img)
        
        # Combine all features
        all_features = {
            'geometric': geometric_features,
            'symmetry': symmetry_features,
            'dots': dot_features,
            'curves': curve_features,
            'topological': topological_features,
            'texture': texture_features,
            'cultural_signature': cultural_signature,
            'complexity_score': self._calculate_complexity_score(
                geometric_features, symmetry_features, curve_features
            ),
            'authenticity_indicators': self._extract_authenticity_indicators(
                processed_img, dot_features, curve_features
            )
        }
        
        return all_features
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for feature extraction."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize image
        gray = cv2.equalizeHist(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def extract_geometric_features(self, image: np.ndarray) -> Dict:
        """
        Extract geometric features including Hu moments, area, perimeter, etc.
        """
        # Find contours
        edges = cv2.Canny(image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._empty_geometric_features()
        
        # Get the largest contour (main pattern)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Basic geometric properties
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Convex hull properties
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Fitted ellipse (if possible)
        try:
            if len(main_contour) >= 5:
                ellipse = cv2.fitEllipse(main_contour)
                ellipse_area = np.pi * (ellipse[1][0]/2) * (ellipse[1][1]/2)
                extent = area / ellipse_area if ellipse_area > 0 else 0
            else:
                extent = 0
        except:
            extent = 0
        
        # Hu moments (shape descriptors)
        moments = cv2.moments(main_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Log transform Hu moments for better numerical stability
        hu_moments_log = []
        for moment in hu_moments:
            if moment != 0:
                hu_moments_log.append(-np.sign(moment) * np.log10(abs(moment)))
            else:
                hu_moments_log.append(0)
        
        # Fourier descriptors
        fourier_descriptors = self._calculate_fourier_descriptors(main_contour)
        
        # Compactness measures
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        rectangularity = area / (w * h) if (w * h) > 0 else 0
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'aspect_ratio': float(aspect_ratio),
            'solidity': float(solidity),
            'extent': float(extent),
            'circularity': float(circularity),
            'rectangularity': float(rectangularity),
            'hu_moments': [float(h) for h in hu_moments_log],
            'fourier_descriptors': fourier_descriptors,
            'contour_count': len(contours),
            'bounding_box': [int(x), int(y), int(w), int(h)]
        }
    
    def calculate_symmetry(self, image: np.ndarray) -> Dict:
        """
        Calculate various symmetry measures for the pattern.
        """
        height, width = image.shape
        center_x, center_y = width // 2, height // 2
        
        # Rotational symmetry (2, 4, 8-fold)
        rotational_symmetries = {}
        for fold in [2, 4, 8]:
            symmetry_score = self._measure_rotational_symmetry(image, fold)
            rotational_symmetries[f'{fold}_fold'] = symmetry_score
        
        # Mirror symmetries
        horizontal_symmetry = self._measure_mirror_symmetry(image, 'horizontal')
        vertical_symmetry = self._measure_mirror_symmetry(image, 'vertical')
        diagonal1_symmetry = self._measure_mirror_symmetry(image, 'diagonal1')
        diagonal2_symmetry = self._measure_mirror_symmetry(image, 'diagonal2')
        
        # Overall symmetry score
        all_symmetries = list(rotational_symmetries.values()) + [
            horizontal_symmetry, vertical_symmetry, diagonal1_symmetry, diagonal2_symmetry
        ]
        overall_symmetry = max(all_symmetries)
        
        # Symmetry axis detection
        symmetry_axes = self._detect_symmetry_axes(image)
        
        return {
            'rotational': rotational_symmetries,
            'mirror_horizontal': float(horizontal_symmetry),
            'mirror_vertical': float(vertical_symmetry),
            'mirror_diagonal1': float(diagonal1_symmetry),
            'mirror_diagonal2': float(diagonal2_symmetry),
            'overall_symmetry': float(overall_symmetry),
            'symmetry_axes': symmetry_axes,
            'is_symmetric': overall_symmetry > self.symmetry_threshold
        }
    
    def detect_dots(self, image: np.ndarray) -> Dict:
        """
        Detect dots in Kolam patterns using Hough Circle Transform and blob detection.
        """
        # Hough Circle Transform for dot detection
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.dot_detection_params['min_distance'],
            param1=self.dot_detection_params['param1'],
            param2=self.dot_detection_params['param2'],
            minRadius=self.dot_detection_params['min_radius'],
            maxRadius=self.dot_detection_params['max_radius']
        )
        
        dots = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            dots = [(int(x), int(y), int(r)) for x, y, r in circles]
        
        # Additional blob detection for irregular dots
        blob_detector = cv2.SimpleBlobDetector_create()
        keypoints = blob_detector.detect(image)
        blob_dots = [(int(kp.pt[0]), int(kp.pt[1]), int(kp.size/2)) for kp in keypoints]
        
        # Combine and deduplicate
        all_dots = self._merge_detections(dots, blob_dots)
        
        # Analyze dot arrangement
        dot_analysis = self._analyze_dot_arrangement(all_dots, image.shape)
        
        return {
            'dot_count': len(all_dots),
            'dot_positions': all_dots,
            'has_dots': len(all_dots) > 0,
            'dot_grid_regularity': dot_analysis['grid_regularity'],
            'average_dot_size': dot_analysis['avg_size'],
            'dot_density': dot_analysis['density'],
            'grid_dimensions': dot_analysis['grid_dims']
        }
    
    def analyze_curves(self, image: np.ndarray) -> Dict:
        """
        Analyze curves and lines in the pattern.
        """
        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._empty_curve_features()
        
        # Analyze each contour
        curve_properties = []
        total_curvature = 0
        total_length = 0
        
        for contour in contours:
            if len(contour) < 5:  # Skip very small contours
                continue
                
            # Calculate curve properties
            length = cv2.arcLength(contour, False)
            curvature = self._calculate_curvature(contour)
            smoothness = self._calculate_smoothness(contour)
            
            curve_properties.append({
                'length': length,
                'curvature': curvature,
                'smoothness': smoothness,
                'is_closed': cv2.isContourConvex(contour)
            })
            
            total_curvature += curvature
            total_length += length
        
        # Line detection using Hough Transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=50, 
            minLineLength=30, maxLineGap=10
        )
        
        line_count = len(lines) if lines is not None else 0
        
        # Curve continuity analysis
        continuity_score = self._analyze_continuity(contours)
        
        # Intersection analysis
        intersection_count = self._count_intersections(contours)
        
        return {
            'curve_count': len(curve_properties),
            'line_count': line_count,
            'total_curve_length': float(total_length),
            'average_curvature': float(total_curvature / len(curve_properties)) if curve_properties else 0,
            'curve_properties': curve_properties,
            'continuity_score': float(continuity_score),
            'intersection_count': intersection_count,
            'has_continuous_path': continuity_score > 0.7
        }
    
    def _extract_topological_features(self, image: np.ndarray) -> Dict:
        """Extract topological properties like genus, holes, connected components."""
        # Binarize image
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary)
        connected_components = num_labels - 1  # Subtract background
        
        # Calculate Euler characteristic and genus
        # Find contours for topology analysis
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Count holes (internal contours)
        holes = 0
        if hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                # If contour has a parent, it's a hole
                if h[3] != -1:  # Has parent
                    holes += 1
        
        # Estimate genus using Euler characteristic
        # χ = V - E + F = 2 - 2g for a surface of genus g
        # For binary images: χ = connected_components - holes
        euler_characteristic = connected_components - holes
        genus = max(0, (2 - euler_characteristic) // 2)
        
        return {
            'connected_components': connected_components,
            'holes': holes,
            'euler_characteristic': euler_characteristic,
            'genus': genus,
            'is_simply_connected': holes == 0
        }
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict:
        """Extract texture features using GLCM and other methods."""
        # Gray Level Co-occurrence Matrix features
        glcm = graycomatrix(
            image, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256, symmetric=True, normed=True
        )
        
        # Calculate GLCM properties
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # Local Binary Pattern (simplified)
        lbp_variance = self._calculate_lbp_variance(image)
        
        return {
            'glcm_contrast': float(contrast),
            'glcm_dissimilarity': float(dissimilarity),
            'glcm_homogeneity': float(homogeneity),
            'glcm_energy': float(energy),
            'glcm_correlation': float(correlation),
            'lbp_variance': float(lbp_variance)
        }
    
    def _extract_cultural_signature(self, image: np.ndarray) -> Dict:
        """Extract cultural pattern signatures specific to Kolam patterns."""
        # Radial analysis from center
        center = (image.shape[1] // 2, image.shape[0] // 2)
        radial_profiles = []
        
        for radius in self.signature_radii:
            profile = self._extract_radial_profile(image, center, radius)
            radial_profiles.append(profile)
        
        # Angular analysis
        angular_profile = self._extract_angular_profile(image, center)
        
        # Pattern density variation
        density_map = self._calculate_density_map(image)
        
        # Traditional motif indicators
        motif_scores = self._detect_traditional_motifs(image)
        
        return {
            'radial_profiles': radial_profiles,
            'angular_profile': angular_profile,
            'density_variation': float(np.std(density_map)),
            'center_density': float(density_map[center[1], center[0]]),
            'traditional_motifs': motif_scores,
            'pattern_regularity': self._measure_pattern_regularity(image)
        }
    
    # Helper methods
    def _empty_geometric_features(self) -> Dict:
        """Return empty geometric features dictionary."""
        return {
            'area': 0.0, 'perimeter': 0.0, 'aspect_ratio': 0.0,
            'solidity': 0.0, 'extent': 0.0, 'circularity': 0.0,
            'rectangularity': 0.0, 'hu_moments': [0.0] * 7,
            'fourier_descriptors': [], 'contour_count': 0,
            'bounding_box': [0, 0, 0, 0]
        }
    
    def _empty_curve_features(self) -> Dict:
        """Return empty curve features dictionary."""
        return {
            'curve_count': 0, 'line_count': 0, 'total_curve_length': 0.0,
            'average_curvature': 0.0, 'curve_properties': [],
            'continuity_score': 0.0, 'intersection_count': 0,
            'has_continuous_path': False
        }
    
    def _calculate_fourier_descriptors(self, contour: np.ndarray, num_descriptors: int = 10) -> List[float]:
        """Calculate Fourier descriptors for shape analysis."""
        if len(contour) < 3:
            return [0.0] * num_descriptors
        
        # Convert contour to complex numbers
        contour_complex = contour[:, 0, 0] + 1j * contour[:, 0, 1]
        
        # Calculate Fourier transform
        fourier_transform = np.fft.fft(contour_complex)
        
        # Get magnitude of first few descriptors (normalized)
        descriptors = np.abs(fourier_transform[:num_descriptors])
        
        # Normalize by first descriptor to achieve translation invariance
        if descriptors[0] > 0:
            descriptors = descriptors / descriptors[0]
        
        return descriptors.tolist()
    
    def _measure_rotational_symmetry(self, image: np.ndarray, fold: int) -> float:
        """Measure rotational symmetry of specified fold."""
        height, width = image.shape
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        angle = 360 / fold
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Calculate similarity between original and rotated
        diff = cv2.absdiff(image, rotated)
        similarity = 1.0 - (np.mean(diff) / 255.0)
        
        return similarity
    
    def _measure_mirror_symmetry(self, image: np.ndarray, axis: str) -> float:
        """Measure mirror symmetry along specified axis."""
        if axis == 'horizontal':
            mirrored = cv2.flip(image, 0)  # Flip vertically
        elif axis == 'vertical':
            mirrored = cv2.flip(image, 1)  # Flip horizontally
        elif axis == 'diagonal1':
            mirrored = cv2.transpose(image)
        elif axis == 'diagonal2':
            mirrored = cv2.flip(cv2.transpose(image), 1)
        else:
            return 0.0
        
        # Calculate similarity
        diff = cv2.absdiff(image, mirrored)
        similarity = 1.0 - (np.mean(diff) / 255.0)
        
        return similarity
    
    def _detect_symmetry_axes(self, image: np.ndarray) -> List[Dict]:
        """Detect potential symmetry axes in the pattern."""
        axes = []
        
        # Test various angles for reflection axes
        height, width = image.shape
        center = (width // 2, height // 2)
        
        for angle in range(0, 180, 10):  # Test every 10 degrees
            # Create reflection matrix
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            
            # Simple reflection test (approximate)
            score = self._test_reflection_axis(image, center, angle)
            
            if score > 0.8:  # High symmetry threshold
                axes.append({
                    'angle': angle,
                    'strength': score,
                    'type': 'reflection'
                })
        
        return axes
    
    def _test_reflection_axis(self, image: np.ndarray, center: Tuple[int, int], angle: float) -> float:
        """Test symmetry along a reflection axis at given angle."""
        # Simplified test - rotate image, flip, rotate back, compare
        height, width = image.shape
        
        # Rotate to make axis horizontal
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Flip horizontally
        flipped = cv2.flip(rotated, 1)
        
        # Rotate back
        rotation_matrix_back = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(flipped, rotation_matrix_back, (width, height))
        
        # Compare with original
        diff = cv2.absdiff(image, result)
        similarity = 1.0 - (np.mean(diff) / 255.0)
        
        return similarity
    
    def _merge_detections(self, dots1: List, dots2: List, threshold: float = 10.0) -> List:
        """Merge two sets of dot detections, removing duplicates."""
        if not dots1:
            return dots2
        if not dots2:
            return dots1
        
        merged = dots1.copy()
        
        for dot2 in dots2:
            x2, y2 = dot2[0], dot2[1]
            is_duplicate = False
            
            for dot1 in dots1:
                x1, y1 = dot1[0], dot1[1]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if distance < threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(dot2)
        
        return merged
    
    def _analyze_dot_arrangement(self, dots: List, image_shape: Tuple[int, int]) -> Dict:
        """Analyze the arrangement and regularity of detected dots."""
        if not dots:
            return {
                'grid_regularity': 0.0,
                'avg_size': 0.0,
                'density': 0.0,
                'grid_dims': (0, 0)
            }
        
        # Extract positions and sizes
        positions = np.array([(dot[0], dot[1]) for dot in dots])
        sizes = [dot[2] if len(dot) > 2 else 5 for dot in dots]
        
        # Calculate density
        image_area = image_shape[0] * image_shape[1]
        density = len(dots) / image_area
        
        # Estimate grid regularity
        if len(positions) > 4:
            # Calculate pairwise distances
            distances = cdist(positions, positions)
            
            # Find most common distance (approximate grid spacing)
            non_zero_distances = distances[distances > 0]
            if len(non_zero_distances) > 0:
                hist, bins = np.histogram(non_zero_distances, bins=20)
                most_common_distance = bins[np.argmax(hist)]
                
                # Count how many distances are close to this spacing
                close_distances = np.sum(
                    np.abs(non_zero_distances - most_common_distance) < most_common_distance * 0.2
                )
                grid_regularity = close_distances / len(non_zero_distances)
            else:
                grid_regularity = 0.0
        else:
            grid_regularity = 0.0
        
        # Estimate grid dimensions
        if len(positions) > 1:
            x_range = np.max(positions[:, 0]) - np.min(positions[:, 0])
            y_range = np.max(positions[:, 1]) - np.min(positions[:, 1])
            
            # Rough estimate based on distribution
            x_unique = len(np.unique(np.round(positions[:, 0] / 10) * 10))
            y_unique = len(np.unique(np.round(positions[:, 1] / 10) * 10))
            grid_dims = (min(x_unique, 20), min(y_unique, 20))
        else:
            grid_dims = (1, 1)
        
        return {
            'grid_regularity': float(grid_regularity),
            'avg_size': float(np.mean(sizes)),
            'density': float(density * 1000000),  # Per million pixels
            'grid_dims': grid_dims
        }
    
    def _calculate_curvature(self, contour: np.ndarray) -> float:
        """Calculate average curvature of a contour."""
        if len(contour) < 3:
            return 0.0
        
        curvatures = []
        points = contour[:, 0, :]  # Remove extra dimension
        
        for i in range(1, len(points) - 1):
            p1, p2, p3 = points[i-1], points[i], points[i+1]
            
            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Calculate angle between vectors
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norms > 0:
                cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                curvatures.append(angle)
        
        return float(np.mean(curvatures)) if curvatures else 0.0
    
    def _calculate_smoothness(self, contour: np.ndarray) -> float:
        """Calculate smoothness of a contour."""
        if len(contour) < 3:
            return 0.0
        
        # Calculate second derivatives (approximation of smoothness)
        points = contour[:, 0, :].astype(float)
        
        # Calculate first derivatives
        first_derivatives = np.diff(points, axis=0)
        
        # Calculate second derivatives
        second_derivatives = np.diff(first_derivatives, axis=0)
        
        # Smoothness is inverse of average second derivative magnitude
        smoothness_values = np.linalg.norm(second_derivatives, axis=1)
        average_roughness = np.mean(smoothness_values)
        
        # Convert to smoothness (0 = rough, 1 = smooth)
        smoothness = 1.0 / (1.0 + average_roughness)
        
        return float(smoothness)
    
    def _analyze_continuity(self, contours: List[np.ndarray]) -> float:
        """Analyze path continuity in the pattern."""
        if not contours:
            return 0.0
        
        total_length = 0
        gap_length = 0
        
        # For each pair of contours, check if they're connected
        for i, contour1 in enumerate(contours):
            if len(contour1) < 2:
                continue
                
            total_length += cv2.arcLength(contour1, False)
            
            for j, contour2 in enumerate(contours[i+1:], i+1):
                if len(contour2) < 2:
                    continue
                
                # Find closest points between contours
                min_distance = float('inf')
                for point1 in contour1[:, 0, :]:
                    for point2 in contour2[:, 0, :]:
                        distance = np.linalg.norm(point1 - point2)
                        min_distance = min(min_distance, distance)
                
                # If contours are close, consider the gap
                if min_distance < 20:  # Threshold for connection
                    gap_length += min_distance
        
        # Continuity score based on ratio of gaps to total length
        if total_length > 0:
            continuity = 1.0 - (gap_length / total_length)
        else:
            continuity = 0.0
        
        return max(0.0, min(1.0, continuity))
    
    def _count_intersections(self, contours: List[np.ndarray]) -> int:
        """Count intersection points between curves."""
        intersection_count = 0
        
        # Simplified intersection counting
        # In practice, would use more sophisticated line intersection algorithms
        for i, contour1 in enumerate(contours):
            for contour2 in contours[i+1:]:
                # Approximate intersection count by checking proximity of points
                for point1 in contour1[::5, 0, :]:  # Sample every 5th point
                    for point2 in contour2[::5, 0, :]:
                        distance = np.linalg.norm(point1 - point2)
                        if distance < 3:  # Very close points suggest intersection
                            intersection_count += 1
        
        return intersection_count
    
    def _calculate_lbp_variance(self, image: np.ndarray) -> float:
        """Calculate Local Binary Pattern variance for texture analysis."""
        # Simplified LBP implementation
        height, width = image.shape
        lbp_values = []
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                center = image[y, x]
                
                # Compare with 8 neighbors
                neighbors = [
                    image[y-1, x-1], image[y-1, x], image[y-1, x+1],
                    image[y, x+1], image[y+1, x+1], image[y+1, x],
                    image[y+1, x-1], image[y, x-1]
                ]
                
                # Create binary pattern
                binary_string = ''
                for neighbor in neighbors:
                    binary_string += '1' if neighbor >= center else '0'
                
                # Convert to decimal
                lbp_value = int(binary_string, 2)
                lbp_values.append(lbp_value)
        
        return float(np.var(lbp_values)) if lbp_values else 0.0
    
    def _extract_radial_profile(self, image: np.ndarray, center: Tuple[int, int], radius: int) -> List[float]:
        """Extract radial intensity profile at given radius from center."""
        profile = []
        
        for angle in range(0, 360, 360 // self.angular_bins):
            rad = np.radians(angle)
            x = int(center[0] + radius * np.cos(rad))
            y = int(center[1] + radius * np.sin(rad))
            
            # Check bounds
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                profile.append(float(image[y, x]))
            else:
                profile.append(0.0)
        
        return profile
    
    def _extract_angular_profile(self, image: np.ndarray, center: Tuple[int, int]) -> List[float]:
        """Extract angular intensity profile from center."""
        profile = []
        max_radius = min(center[0], center[1], 
                        image.shape[1] - center[0], 
                        image.shape[0] - center[1])
        
        for angle in range(0, 360, 360 // self.angular_bins):
            rad = np.radians(angle)
            
            # Sample along the ray
            intensities = []
            for r in range(1, max_radius, max(1, max_radius // 20)):
                x = int(center[0] + r * np.cos(rad))
                y = int(center[1] + r * np.sin(rad))
                
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    intensities.append(image[y, x])
            
            # Average intensity along this ray
            if intensities:
                profile.append(float(np.mean(intensities)))
            else:
                profile.append(0.0)
        
        return profile
    
    def _calculate_density_map(self, image: np.ndarray, window_size: int = 20) -> np.ndarray:
        """Calculate local density map of the pattern."""
        # Use binary threshold
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Calculate local density using convolution
        kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size * window_size)
        density_map = cv2.filter2D(binary.astype(np.float32), -1, kernel)
        
        return density_map
    
    def _detect_traditional_motifs(self, image: np.ndarray) -> Dict[str, float]:
        """Detect traditional Kolam motifs in the pattern."""
        motif_scores = {}
        
        # Template matching for common motifs (simplified)
        # In practice, would use learned templates or feature detectors
        
        # Circular motifs (flowers, lotus)
        circular_score = self._detect_circular_motifs(image)
        motif_scores['circular'] = circular_score
        
        # Leaf-like motifs
        leaf_score = self._detect_leaf_motifs(image)
        motif_scores['leaf'] = leaf_score
        
        # Geometric patterns (squares, triangles)
        geometric_score = self._detect_geometric_motifs(image)
        motif_scores['geometric'] = geometric_score
        
        # Star-like patterns
        star_score = self._detect_star_motifs(image)
        motif_scores['star'] = star_score
        
        return motif_scores
    
    def _detect_circular_motifs(self, image: np.ndarray) -> float:
        """Detect circular/flower-like motifs."""
        # Use Hough circle detection as proxy for circular motifs
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        circle_count = len(circles[0]) if circles is not None else 0
        
        # Normalize score based on image size
        image_area = image.shape[0] * image.shape[1]
        normalized_score = min(1.0, circle_count * 10000 / image_area)
        
        return float(normalized_score)
    
    def _detect_leaf_motifs(self, image: np.ndarray) -> float:
        """Detect leaf-like motifs using contour analysis."""
        edges = cv2.Canny(image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        leaf_like_count = 0
        
        for contour in contours:
            if len(contour) < 5:
                continue
            
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area == 0 or perimeter == 0:
                continue
            
            # Leaf-like shapes tend to have specific aspect ratios and solidity
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                
                # Leaf-like criteria
                if 1.5 < aspect_ratio < 4.0:  # Elongated but not too thin
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    if 0.6 < solidity < 0.9:  # Somewhat concave (leaf-like)
                        leaf_like_count += 1
        
        # Normalize score
        image_area = image.shape[0] * image.shape[1]
        normalized_score = min(1.0, leaf_like_count * 20000 / image_area)
        
        return float(normalized_score)
    
    def _detect_geometric_motifs(self, image: np.ndarray) -> float:
        """Detect geometric motifs (squares, triangles, etc.)."""
        edges = cv2.Canny(image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        geometric_count = 0
        
        for contour in contours:
            if len(contour) < 3:
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Count shapes with 3-8 sides (geometric)
            if 3 <= len(approx) <= 8:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    geometric_count += 1
        
        # Normalize score
        image_area = image.shape[0] * image.shape[1]
        normalized_score = min(1.0, geometric_count * 15000 / image_area)
        
        return float(normalized_score)
    
    def _detect_star_motifs(self, image: np.ndarray) -> float:
        """Detect star-like patterns using corner detection."""
        # Use Harris corner detection to find star-like patterns
        gray = image.astype(np.float32)
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        
        # Threshold corner response
        corner_threshold = 0.01 * corners.max()
        corner_points = np.where(corners > corner_threshold)
        
        # Count potential star centers (high corner response)
        star_candidates = len(corner_points[0])
        
        # Additional check: look for radial symmetry around corner points
        star_count = 0
        for i in range(min(star_candidates, 50)):  # Limit to prevent slowdown
            y, x = corner_points[0][i], corner_points[1][i]
            
            # Check for radial pattern around this point
            radial_score = self._check_radial_pattern(image, (x, y))
            if radial_score > 0.7:
                star_count += 1
        
        # Normalize score
        image_area = image.shape[0] * image.shape[1]
        normalized_score = min(1.0, star_count * 25000 / image_area)
        
        return float(normalized_score)
    
    def _check_radial_pattern(self, image: np.ndarray, center: Tuple[int, int], num_rays: int = 8) -> float:
        """Check for radial pattern around a point."""
        if (center[0] < 10 or center[0] >= image.shape[1] - 10 or 
            center[1] < 10 or center[1] >= image.shape[0] - 10):
            return 0.0
        
        ray_intensities = []
        
        for i in range(num_rays):
            angle = 2 * np.pi * i / num_rays
            
            # Sample along ray
            intensities = []
            for r in range(5, 20):  # Sample from radius 5 to 20
                x = int(center[0] + r * np.cos(angle))
                y = int(center[1] + r * np.sin(angle))
                
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    intensities.append(image[y, x])
            
            if intensities:
                ray_intensities.append(np.mean(intensities))
        
        if len(ray_intensities) < num_rays:
            return 0.0
        
        # Check for variation (star patterns have alternating high/low rays)
        variation = np.std(ray_intensities)
        mean_intensity = np.mean(ray_intensities)
        
        if mean_intensity > 0:
            normalized_variation = variation / mean_intensity
            return min(1.0, normalized_variation * 2)
        else:
            return 0.0
    
    def _measure_pattern_regularity(self, image: np.ndarray) -> float:
        """Measure overall pattern regularity."""
        # Calculate local variance to measure regularity
        # Regular patterns have consistent local structure
        
        # Divide image into blocks and calculate variance of each block
        block_size = 20
        height, width = image.shape
        
        block_variances = []
        
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                block = image[y:y+block_size, x:x+block_size]
                block_variance = np.var(block.astype(np.float32))
                block_variances.append(block_variance)
        
        if not block_variances:
            return 0.0
        
        # Regularity is inverse of variance in block variances
        # Regular patterns have similar variance across blocks
        overall_variance = np.var(block_variances)
        mean_variance = np.mean(block_variances)
        
        if mean_variance > 0:
            regularity = 1.0 / (1.0 + overall_variance / mean_variance)
        else:
            regularity = 0.0
        
        return float(regularity)
    
    def _calculate_complexity_score(self, geometric_features: Dict, 
                                  symmetry_features: Dict, 
                                  curve_features: Dict) -> float:
        """Calculate overall pattern complexity score."""
        
        # Extract relevant metrics
        contour_count = geometric_features.get('contour_count', 0)
        curve_count = curve_features.get('curve_count', 0)
        intersection_count = curve_features.get('intersection_count', 0)
        overall_symmetry = symmetry_features.get('overall_symmetry', 0)
        
        # Normalize metrics
        normalized_contours = min(1.0, contour_count / 20.0)
        normalized_curves = min(1.0, curve_count / 15.0)
        normalized_intersections = min(1.0, intersection_count / 10.0)
        symmetry_deviation = 1.0 - overall_symmetry
        
        # Calculate weighted complexity
        complexity = (
            self.complexity_weights['contour_count'] * normalized_contours +
            self.complexity_weights['curvature_variation'] * normalized_curves +
            self.complexity_weights['intersection_points'] * normalized_intersections +
            self.complexity_weights['symmetry_deviation'] * symmetry_deviation
        )
        
        return float(min(1.0, complexity))
    
    def _extract_authenticity_indicators(self, image: np.ndarray, 
                                       dot_features: Dict, 
                                       curve_features: Dict) -> Dict[str, float]:
        """Extract indicators of cultural authenticity."""
        
        indicators = {}
        
        # Traditional dot grid presence
        if dot_features.get('has_dots', False):
            grid_regularity = dot_features.get('dot_grid_regularity', 0)
            indicators['traditional_dot_grid'] = float(grid_regularity)
        else:
            indicators['traditional_dot_grid'] = 0.0
        
        # Continuous path indicator (important for many Kolam types)
        indicators['continuous_path'] = float(curve_features.get('continuity_score', 0))
        
        # Symmetry adherence (very important for Kolam)
        symmetry_score = self._calculate_overall_symmetry_adherence(image)
        indicators['symmetry_adherence'] = float(symmetry_score)
        
        # Mathematical precision (clean lines, proper curves)
        precision_score = self._calculate_mathematical_precision(curve_features)
        indicators['mathematical_precision'] = float(precision_score)
        
        # Traditional proportions
        proportion_score = self._check_traditional_proportions(image, dot_features)
        indicators['traditional_proportions'] = float(proportion_score)
        
        return indicators
    
    def _calculate_overall_symmetry_adherence(self, image: np.ndarray) -> float:
        """Calculate adherence to symmetry principles."""
        # Test multiple types of symmetry
        symmetries = []
        
        # Test rotational symmetries
        for fold in [2, 4, 8]:
            score = self._measure_rotational_symmetry(image, fold)
            symmetries.append(score)
        
        # Test mirror symmetries
        for axis in ['horizontal', 'vertical', 'diagonal1', 'diagonal2']:
            score = self._measure_mirror_symmetry(image, axis)
            symmetries.append(score)
        
        # Return the highest symmetry score
        return max(symmetries) if symmetries else 0.0
    
    def _calculate_mathematical_precision(self, curve_features: Dict) -> float:
        """Calculate mathematical precision of curves and lines."""
        curve_properties = curve_features.get('curve_properties', [])
        
        if not curve_properties:
            return 0.0
        
        # Average smoothness indicates precision
        smoothness_values = [prop.get('smoothness', 0) for prop in curve_properties]
        average_smoothness = np.mean(smoothness_values) if smoothness_values else 0
        
        # Line count also indicates precision (straight elements)
        line_count = curve_features.get('line_count', 0)
        curve_count = curve_features.get('curve_count', 1)  # Avoid division by zero
        
        line_ratio = min(1.0, line_count / max(1, curve_count))
        
        # Combine smoothness and line precision
        precision = 0.7 * average_smoothness + 0.3 * line_ratio
        
        return precision
    
    def _check_traditional_proportions(self, image: np.ndarray, dot_features: Dict) -> float:
        """Check adherence to traditional Kolam proportions."""
        height, width = image.shape
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Traditional Kolam often have square or slightly rectangular proportions
        ideal_ratios = [1.0, 1.2, 0.8, 1.5, 2.0/3.0]  # Common traditional ratios
        
        # Find closest ideal ratio
        ratio_deviations = [abs(aspect_ratio - ideal) for ideal in ideal_ratios]
        min_deviation = min(ratio_deviations)
        
        # Score based on closeness to ideal ratio
        ratio_score = max(0.0, 1.0 - min_deviation)
        
        # Check dot spacing regularity (if dots present)
        if dot_features.get('has_dots', False):
            grid_regularity = dot_features.get('dot_grid_regularity', 0)
            spacing_score = grid_regularity
        else:
            spacing_score = 1.0  # Not applicable for non-dot patterns
        
        # Combine ratio and spacing scores
        proportion_score = 0.6 * ratio_score + 0.4 * spacing_score
        
        return proportion_score


# Utility functions for external use
def extract_features_from_file(image_path: str) -> Dict:
    """
    Extract features from an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing extracted features
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        extractor = FeatureExtractor()
        features = extractor.extract_all_features(image)
        
        return features
    except Exception as e:
        return {
            'error': str(e),
            'features_extracted': False
        }

def extract_features_batch(image_paths: List[str]) -> List[Dict]:
    """
    Extract features from multiple image files.
    
    Args:
        image_paths: List of paths to image files
        
    Returns:
        List of feature dictionaries
    """
    results = []
    extractor = FeatureExtractor()
    
    for image_path in image_paths:
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                results.append({
                    'image_path': image_path,
                    'error': f"Could not load image from {image_path}",
                    'features_extracted': False
                })
                continue
            
            features = extractor.extract_all_features(image)
            features['image_path'] = image_path
            features['features_extracted'] = True
            
            results.append(features)
            
        except Exception as e:
            results.append({
                'image_path': image_path,
                'error': str(e),
                'features_extracted': False
            })
    
    return results

def save_features_to_json(features: Dict, output_path: str) -> bool:
    """
    Save extracted features to JSON file.
    
    Args:
        features: Feature dictionary
        output_path: Path to save JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json_features = convert_numpy(features)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_features, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"Error saving features to JSON: {e}")
        return False