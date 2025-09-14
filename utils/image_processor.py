import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Union
import math
from scipy import ndimage
from skimage import filters, morphology, restoration
from skimage.segmentation import watershed
from skimage.feature import peak_local_maxima
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Core image processing utilities using OpenCV for Kolam pattern analysis.
    Provides preprocessing, edge detection, contour analysis, and enhancement methods.
    """
    
    def __init__(self):
        # Canny edge detection thresholds
        self.canny_thresholds = {
            'low': 50,
            'high': 150,
            'aperture_size': 3,
            'l2_gradient': False
        }
        
        # Morphological kernel sizes
        self.kernel_size = {
            'small': (3, 3),
            'medium': (5, 5),
            'large': (7, 7),
            'xlarge': (9, 9)
        }
        
        # Gaussian blur parameters
        self.blur_params = {
            'kernel_size': (5, 5),
            'sigma_x': 1.0,
            'sigma_y': 1.0
        }
        
        # Noise reduction parameters
        self.denoise_params = {
            'h': 10,
            'template_window_size': 7,
            'search_window_size': 21
        }
        
        # Histogram equalization parameters
        self.clahe_params = {
            'clip_limit': 2.0,
            'tile_grid_size': (8, 8)
        }
        
    def preprocess(self, image: np.ndarray, 
                   resize_target: Optional[Tuple[int, int]] = None,
                   enhance_contrast: bool = True,
                   reduce_noise: bool = True,
                   normalize: bool = True) -> np.ndarray:
        """
        Comprehensive image preprocessing pipeline for Kolam analysis.
        
        Args:
            image: Input image (color or grayscale)
            resize_target: Target size (width, height) for resizing
            enhance_contrast: Whether to apply contrast enhancement
            reduce_noise: Whether to apply noise reduction
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed grayscale image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                processed = image.copy()
            
            # Resize if specified
            if resize_target:
                processed = cv2.resize(processed, resize_target, interpolation=cv2.INTER_LANCZOS4)
                logger.info(f"Resized image to {resize_target}")
            
            # Noise reduction
            if reduce_noise:
                processed = self.remove_noise(processed)
                logger.info("Applied noise reduction")
            
            # Contrast enhancement
            if enhance_contrast:
                processed = self.enhance_contrast(processed)
                logger.info("Enhanced contrast")
            
            # Normalization
            if normalize:
                processed = self._normalize_image(processed)
                logger.info("Normalized image")
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def detect_edges(self, image: np.ndarray, 
                    method: str = 'canny',
                    auto_threshold: bool = False,
                    **kwargs) -> np.ndarray:
        """
        Detect edges using various methods optimized for Kolam patterns.
        
        Args:
            image: Input grayscale image
            method: Edge detection method ('canny', 'sobel', 'laplacian', 'scharr')
            auto_threshold: Whether to automatically determine thresholds
            **kwargs: Additional parameters for edge detection
            
        Returns:
            Binary edge image
        """
        if method == 'canny':
            return self._canny_edges(image, auto_threshold, **kwargs)
        elif method == 'sobel':
            return self._sobel_edges(image, **kwargs)
        elif method == 'laplacian':
            return self._laplacian_edges(image, **kwargs)
        elif method == 'scharr':
            return self._scharr_edges(image, **kwargs)
        else:
            logger.warning(f"Unknown edge detection method: {method}. Using Canny.")
            return self._canny_edges(image, auto_threshold, **kwargs)
    
    def find_contours(self, image: np.ndarray, 
                     mode: int = cv2.RETR_EXTERNAL,
                     method: int = cv2.CHAIN_APPROX_SIMPLE,
                     min_area: float = 100.0,
                     max_area: Optional[float] = None) -> List[np.ndarray]:
        """
        Find and filter contours suitable for Kolam analysis.
        
        Args:
            image: Binary input image
            mode: Contour retrieval mode
            method: Contour approximation method
            min_area: Minimum contour area to keep
            max_area: Maximum contour area to keep (None for no limit)
            
        Returns:
            List of filtered contours
        """
        try:
            # Ensure binary image
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, hierarchy = cv2.findContours(binary, mode, method)
            
            # Filter contours by area
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    if max_area is None or area <= max_area:
                        filtered_contours.append(contour)
            
            logger.info(f"Found {len(filtered_contours)} valid contours out of {len(contours)} total")
            return filtered_contours
            
        except Exception as e:
            logger.error(f"Error finding contours: {e}")
            return []
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Enhance image contrast using various methods.
        
        Args:
            image: Input grayscale image
            method: Enhancement method ('clahe', 'hist_eq', 'gamma', 'adaptive')
            
        Returns:
            Contrast-enhanced image
        """
        if method == 'clahe':
            return self._clahe_enhancement(image)
        elif method == 'hist_eq':
            return cv2.equalizeHist(image)
        elif method == 'gamma':
            return self._gamma_correction(image, gamma=1.2)
        elif method == 'adaptive':
            return self._adaptive_enhancement(image)
        else:
            logger.warning(f"Unknown contrast method: {method}. Using CLAHE.")
            return self._clahe_enhancement(image)
    
    def remove_noise(self, image: np.ndarray, method: str = 'non_local_means') -> np.ndarray:
        """
        Remove noise from image using various denoising methods.
        
        Args:
            image: Input grayscale image
            method: Denoising method ('non_local_means', 'gaussian', 'bilateral', 'median')
            
        Returns:
            Denoised image
        """
        try:
            if method == 'non_local_means':
                return cv2.fastNlMeansDenoising(
                    image,
                    h=self.denoise_params['h'],
                    templateWindowSize=self.denoise_params['template_window_size'],
                    searchWindowSize=self.denoise_params['search_window_size']
                )
            elif method == 'gaussian':
                return cv2.GaussianBlur(image, self.blur_params['kernel_size'], 
                                      self.blur_params['sigma_x'])
            elif method == 'bilateral':
                return cv2.bilateralFilter(image, 9, 75, 75)
            elif method == 'median':
                return cv2.medianBlur(image, 5)
            else:
                logger.warning(f"Unknown denoising method: {method}. Using Non-local means.")
                return cv2.fastNlMeansDenoising(image)
                
        except Exception as e:
            logger.error(f"Error in noise removal: {e}")
            return image
    
    def geometric_transformations(self, image: np.ndarray, 
                                transform_type: str,
                                **kwargs) -> np.ndarray:
        """
        Apply geometric transformations for image correction and augmentation.
        
        Args:
            image: Input image
            transform_type: Type of transformation 
                          ('rotate', 'scale', 'translate', 'perspective', 'affine')
            **kwargs: Transformation parameters
            
        Returns:
            Transformed image
        """
        height, width = image.shape[:2]
        
        if transform_type == 'rotate':
            angle = kwargs.get('angle', 0)
            center = kwargs.get('center', (width // 2, height // 2))
            scale = kwargs.get('scale', 1.0)
            
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            return cv2.warpAffine(image, rotation_matrix, (width, height))
            
        elif transform_type == 'scale':
            scale_x = kwargs.get('scale_x', 1.0)
            scale_y = kwargs.get('scale_y', 1.0)
            
            new_width = int(width * scale_x)
            new_height = int(height * scale_y)
            return cv2.resize(image, (new_width, new_height))
            
        elif transform_type == 'translate':
            tx = kwargs.get('tx', 0)
            ty = kwargs.get('ty', 0)
            
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            return cv2.warpAffine(image, translation_matrix, (width, height))
            
        elif transform_type == 'perspective':
            src_points = kwargs.get('src_points')
            dst_points = kwargs.get('dst_points')
            
            if src_points is not None and dst_points is not None:
                perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                return cv2.warpPerspective(image, perspective_matrix, (width, height))
            else:
                logger.error("Perspective transformation requires src_points and dst_points")
                return image
                
        elif transform_type == 'affine':
            src_points = kwargs.get('src_points')
            dst_points = kwargs.get('dst_points')
            
            if src_points is not None and dst_points is not None:
                affine_matrix = cv2.getAffineTransform(src_points, dst_points)
                return cv2.warpAffine(image, affine_matrix, (width, height))
            else:
                logger.error("Affine transformation requires src_points and dst_points")
                return image
        
        else:
            logger.error(f"Unknown transformation type: {transform_type}")
            return image
    
    def morphological_operations(self, image: np.ndarray, 
                               operation: str,
                               kernel_type: str = 'ellipse',
                               kernel_size: str = 'medium',
                               iterations: int = 1) -> np.ndarray:
        """
        Apply morphological operations for pattern cleanup.
        
        Args:
            image: Binary input image
            operation: Morphological operation 
                      ('erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'blackhat')
            kernel_type: Kernel shape ('rect', 'ellipse', 'cross')
            kernel_size: Kernel size ('small', 'medium', 'large', 'xlarge')
            iterations: Number of iterations
            
        Returns:
            Processed binary image
        """
        # Get kernel
        kernel = self._get_morphological_kernel(kernel_type, kernel_size)
        
        # Apply operation
        if operation == 'erode':
            return cv2.erode(image, kernel, iterations=iterations)
        elif operation == 'dilate':
            return cv2.dilate(image, kernel, iterations=iterations)
        elif operation == 'open':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'close':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif operation == 'gradient':
            return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        elif operation == 'tophat':
            return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        elif operation == 'blackhat':
            return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        else:
            logger.error(f"Unknown morphological operation: {operation}")
            return image
    
    def segment_pattern(self, image: np.ndarray, 
                       method: str = 'watershed',
                       **kwargs) -> Tuple[np.ndarray, int]:
        """
        Segment Kolam pattern into distinct regions.
        
        Args:
            image: Input grayscale image
            method: Segmentation method ('watershed', 'kmeans', 'threshold', 'region_growing')
            **kwargs: Method-specific parameters
            
        Returns:
            Tuple of (segmented image, number of segments)
        """
        if method == 'watershed':
            return self._watershed_segmentation(image, **kwargs)
        elif method == 'kmeans':
            return self._kmeans_segmentation(image, **kwargs)
        elif method == 'threshold':
            return self._threshold_segmentation(image, **kwargs)
        elif method == 'region_growing':
            return self._region_growing_segmentation(image, **kwargs)
        else:
            logger.error(f"Unknown segmentation method: {method}")
            return image, 1
    
    def detect_lines(self, image: np.ndarray, 
                    method: str = 'hough_probabilistic',
                    **kwargs) -> List[np.ndarray]:
        """
        Detect lines in the image using various methods.
        
        Args:
            image: Binary input image
            method: Line detection method ('hough', 'hough_probabilistic', 'lsd')
            **kwargs: Method-specific parameters
            
        Returns:
            List of detected lines
        """
        if method == 'hough':
            return self._hough_lines(image, **kwargs)
        elif method == 'hough_probabilistic':
            return self._hough_lines_p(image, **kwargs)
        elif method == 'lsd':
            return self._lsd_lines(image, **kwargs)
        else:
            logger.error(f"Unknown line detection method: {method}")
            return []
    
    def detect_circles(self, image: np.ndarray, **kwargs) -> List[Tuple[int, int, int]]:
        """
        Detect circular patterns (dots) in Kolam images.
        
        Args:
            image: Input grayscale image
            **kwargs: Parameters for circle detection
            
        Returns:
            List of circles as (x, y, radius) tuples
        """
        # Default parameters
        dp = kwargs.get('dp', 1)
        min_dist = kwargs.get('min_dist', 20)
        param1 = kwargs.get('param1', 50)
        param2 = kwargs.get('param2', 30)
        min_radius = kwargs.get('min_radius', 5)
        max_radius = kwargs.get('max_radius', 50)
        
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
            param1=param1, param2=param2, 
            minRadius=min_radius, maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return [(int(x), int(y), int(r)) for x, y, r in circles]
        else:
            return []
    
    def skeletonize(self, image: np.ndarray, method: str = 'zhang_suen') -> np.ndarray:
        """
        Skeletonize binary image to extract pattern structure.
        
        Args:
            image: Binary input image
            method: Skeletonization method ('zhang_suen', 'medial_axis')
            
        Returns:
            Skeletonized binary image
        """
        # Ensure binary
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        
        if method == 'zhang_suen':
            # OpenCV doesn't have Zhang-Suen, use morphological approach
            skeleton = np.zeros(binary.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            
            while True:
                eroded = cv2.erode(binary, element)
                opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
                subset = cv2.subtract(eroded, opened)
                skeleton = cv2.bitwise_or(skeleton, subset)
                binary = eroded.copy()
                
                if cv2.countNonZero(binary) == 0:
                    break
            
            return skeleton
            
        elif method == 'medial_axis':
            # Use distance transform for medial axis
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            
            # Find local maxima
            local_maxima = peak_local_maxima(dist_transform, min_distance=3, threshold_abs=1)
            
            skeleton = np.zeros_like(binary)
            for peak in local_maxima:
                skeleton[peak[0], peak[1]] = 255
                
            return skeleton
        
        else:
            logger.error(f"Unknown skeletonization method: {method}")
            return image
    
    def correct_perspective(self, image: np.ndarray, 
                          corners: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Correct perspective distortion in Kolam images.
        
        Args:
            image: Input image
            corners: Optional corners for perspective correction
                    If None, attempts automatic detection
            
        Returns:
            Perspective-corrected image
        """
        if corners is None:
            corners = self._detect_document_corners(image)
        
        if corners is None:
            logger.warning("Could not detect corners for perspective correction")
            return image
        
        # Define target rectangle
        height, width = image.shape[:2]
        target_corners = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform
        perspective_matrix = cv2.getPerspectiveTransform(corners, target_corners)
        
        # Apply transformation
        corrected = cv2.warpPerspective(image, perspective_matrix, (width, height))
        
        return corrected
    
    # Private helper methods
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values to 0-255 range."""
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
    
    def _canny_edges(self, image: np.ndarray, auto_threshold: bool, **kwargs) -> np.ndarray:
        """Apply Canny edge detection."""
        if auto_threshold:
            # Automatic threshold selection using Otsu's method
            high_thresh, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            low_thresh = 0.5 * high_thresh
        else:
            low_thresh = kwargs.get('low_thresh', self.canny_thresholds['low'])
            high_thresh = kwargs.get('high_thresh', self.canny_thresholds['high'])
        
        aperture_size = kwargs.get('aperture_size', self.canny_thresholds['aperture_size'])
        l2_gradient = kwargs.get('l2_gradient', self.canny_thresholds['l2_gradient'])
        
        return cv2.Canny(image, low_thresh, high_thresh, 
                        apertureSize=aperture_size, L2gradient=l2_gradient)
    
    def _sobel_edges(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Sobel edge detection."""
        ksize = kwargs.get('ksize', 3)
        
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(magnitude / magnitude.max() * 255)
        
        # Threshold to get binary edges
        threshold = kwargs.get('threshold', 50)
        _, binary_edges = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
        
        return binary_edges
    
    def _laplacian_edges(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Laplacian edge detection."""
        ksize = kwargs.get('ksize', 3)
        
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Threshold to get binary edges
        threshold = kwargs.get('threshold', 50)
        _, binary_edges = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)
        
        return binary_edges
    
    def _scharr_edges(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Scharr edge detection."""
        grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(magnitude / magnitude.max() * 255)
        
        # Threshold to get binary edges
        threshold = kwargs.get('threshold', 50)
        _, binary_edges = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
        
        return binary_edges
    
    def _clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_params['clip_limit'],
            tileGridSize=self.clahe_params['tile_grid_size']
        )
        return clahe.apply(image)
    
    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction."""
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype(np.uint8)
        
        return cv2.LUT(image, table)
    
    def _adaptive_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive histogram equalization."""
        return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(image)
    
    def _get_morphological_kernel(self, kernel_type: str, size: str) -> np.ndarray:
        """Get morphological kernel of specified type and size."""
        kernel_size = self.kernel_size[size]
        
        if kernel_type == 'rect':
            return cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        elif kernel_type == 'ellipse':
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        elif kernel_type == 'cross':
            return cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
        else:
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    def _watershed_segmentation(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, int]:
        """Watershed segmentation."""
        # Apply threshold
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Find sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add 1 to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        if len(image.shape) == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image
            
        markers = cv2.watershed(image_color, markers)
        
        # Count segments
        num_segments = len(np.unique(markers)) - 1  # Subtract background
        
        return markers.astype(np.uint8), num_segments
    
    def _kmeans_segmentation(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, int]:
        """K-means segmentation."""
        k = kwargs.get('k', 3)
        
        # Reshape image to a 2D array of pixels
        data = image.reshape((-1, 1))
        data = np.float32(data)
        
        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and reshape to original image shape
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        segmented = segmented.reshape(image.shape)
        
        return segmented, k
    
    def _threshold_segmentation(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, int]:
        """Multi-level threshold segmentation."""
        levels = kwargs.get('levels', 3)
        
        # Calculate thresholds
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        
        # Simple multi-level thresholding
        thresholds = []
        for i in range(1, levels):
            threshold = i * 255 // levels
            thresholds.append(threshold)
        
        # Apply thresholds
        segmented = np.zeros_like(image)
        for i, thresh in enumerate(thresholds):
            if i == 0:
                segmented[image <= thresh] = i * (255 // (levels - 1))
            else:
                segmented[(image > thresholds[i-1]) & (image <= thresh)] = i * (255 // (levels - 1))
        
        # Handle remaining pixels
        if thresholds:
            segmented[image > thresholds[-1]] = 255
        
        return segmented, levels
    
    def _region_growing_segmentation(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, int]:
        """Simple region growing segmentation."""
        threshold = kwargs.get('threshold', 10)
        seeds = kwargs.get('seeds', [(image.shape[0]//2, image.shape[1]//2)])
        
        segmented = np.zeros_like(image)
        segment_id = 1
        
        for seed in seeds:
            if segmented[seed] == 0:  # Not yet segmented
                self._region_grow(image, segmented, seed, segment_id, threshold)
                segment_id += 1
        
        return segmented, segment_id - 1
    
    def _region_grow(self, image: np.ndarray, segmented: np.ndarray, 
                    seed: Tuple[int, int], segment_id: int, threshold: int):
        """Helper function for region growing."""
        height, width = image.shape
        stack = [seed]
        seed_value = image[seed]
        
        while stack:
            current = stack.pop()
            y, x = current
            
            if (0 <= y < height and 0 <= x < width and 
                segmented[y, x] == 0 and 
                abs(int(image[y, x]) - int(seed_value)) <= threshold):
                
                segmented[y, x] = segment_id
                
                # Add neighbors to stack
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_y, new_x = y + dy, x + dx
                    if (0 <= new_y < height and 0 <= new_x < width):
                        stack.append((new_y, new_x))
    
    def _hough_lines(self, image: np.ndarray, **kwargs) -> List[np.ndarray]:
        """Standard Hough line detection."""
        rho = kwargs.get('rho', 1)
        theta = kwargs.get('theta', np.pi / 180)
        threshold = kwargs.get('threshold', 100)
        
        lines = cv2.HoughLines(image, rho, theta, threshold)
        
        if lines is not None:
            return [line[0] for line in lines]
        else:
            return []
    
    def _hough_lines_p(self, image: np.ndarray, **kwargs) -> List[np.ndarray]:
        """Probabilistic Hough line detection."""
        rho = kwargs.get('rho', 1)
        theta = kwargs.get('theta', np.pi / 180)
        threshold = kwargs.get('threshold', 50)
        min_line_length = kwargs.get('min_line_length', 30)
        max_line_gap = kwargs.get('max_line_gap', 10)
        
        lines = cv2.HoughLinesP(image, rho, theta, threshold, 
                               minLineLength=min_line_length, 
                               maxLineGap=max_line_gap)
        
        if lines is not None:
            return [line[0] for line in lines]
        else:
            return []
    
    def _lsd_lines(self, image: np.ndarray, **kwargs) -> List[np.ndarray]:
        """Line Segment Detector (LSD) for line detection."""
        try:
            lsd = cv2.createLineSegmentDetector()
            lines, _, _, _ = lsd.detect(image)
            
            if lines is not None:
                # Convert to format similar to HoughLinesP
                result_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    result_lines.append([int(x1), int(y1), int(x2), int(y2)])
                return result_lines
            else:
                return []
        except Exception as e:
            logger.error(f"Error in LSD line detection: {e}")
            return []
    
    def _detect_document_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect corners of a document/pattern for perspective correction."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find largest contour (assuming it's the document)
            if not contours:
                return None
                
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # If we found a quadrilateral, return the corners
            if len(approx) == 4:
                return np.array([point[0] for point in approx], dtype=np.float32)
            else:
                # Fallback: use corner detection
                corners = cv2.goodFeaturesToTrack(
                    gray, maxCorners=4, qualityLevel=0.01, minDistance=50
                )
                
                if corners is not None and len(corners) >= 4:
                    # Sort corners to get consistent order
                    corners = corners.reshape(-1, 2)
                    corners = self._sort_corners(corners[:4])
                    return corners.astype(np.float32)
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error detecting document corners: {e}")
            return None
    
    def _sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """Sort corners in clockwise order starting from top-left."""
        # Calculate centroid
        centroid = np.mean(corners, axis=0)
        
        # Calculate angles from centroid
        angles = []
        for corner in corners:
            angle = np.arctan2(corner[1] - centroid[1], corner[0] - centroid[0])
            angles.append(angle)
        
        # Sort by angle
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]
        
        return sorted_corners
    
    def analyze_symmetry(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze various types of symmetry in the image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Dictionary containing symmetry scores
        """
        symmetry_scores = {}
        
        try:
            # Ensure grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            height, width = gray.shape
            
            # Vertical symmetry (left-right)
            left_half = gray[:, :width//2]
            right_half = cv2.flip(gray[:, width//2:], 1)
            
            # Resize to match if odd width
            if left_half.shape[1] != right_half.shape[1]:
                min_width = min(left_half.shape[1], right_half.shape[1])
                left_half = left_half[:, :min_width]
                right_half = right_half[:, :min_width]
            
            vertical_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
            symmetry_scores['vertical'] = max(0, 1 - vertical_diff / 255.0)
            
            # Horizontal symmetry (top-bottom)
            top_half = gray[:height//2, :]
            bottom_half = cv2.flip(gray[height//2:, :], 0)
            
            # Resize to match if odd height
            if top_half.shape[0] != bottom_half.shape[0]:
                min_height = min(top_half.shape[0], bottom_half.shape[0])
                top_half = top_half[:min_height, :]
                bottom_half = bottom_half[:min_height, :]
            
            horizontal_diff = np.mean(np.abs(top_half.astype(float) - bottom_half.astype(float)))
            symmetry_scores['horizontal'] = max(0, 1 - horizontal_diff / 255.0)
            
            # Rotational symmetry (180 degrees)
            rotated_180 = cv2.rotate(gray, cv2.ROTATE_180)
            rotation_diff = np.mean(np.abs(gray.astype(float) - rotated_180.astype(float)))
            symmetry_scores['rotational_180'] = max(0, 1 - rotation_diff / 255.0)
            
            # 4-fold rotational symmetry
            rotated_90 = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
            rotation_90_diff = np.mean(np.abs(gray.astype(float) - rotated_90.astype(float)))
            symmetry_scores['rotational_90'] = max(0, 1 - rotation_90_diff / 255.0)
            
            # Overall symmetry score
            symmetry_scores['overall'] = np.mean(list(symmetry_scores.values()))
            
        except Exception as e:
            logger.error(f"Error analyzing symmetry: {e}")
            # Return default scores
            symmetry_scores = {
                'vertical': 0.0,
                'horizontal': 0.0,
                'rotational_180': 0.0,
                'rotational_90': 0.0,
                'overall': 0.0
            }
        
        return symmetry_scores
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features from the image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Dictionary containing texture features
        """
        features = {}
        
        try:
            # Ensure grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Local Binary Pattern (simplified version)
            lbp = self._simple_lbp(gray)
            features['lbp_uniformity'] = self._calculate_uniformity(lbp)
            
            # Gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features['gradient_mean'] = np.mean(gradient_magnitude)
            features['gradient_std'] = np.std(gradient_magnitude)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            features['edge_density'] = edge_pixels / total_pixels
            
            # Intensity statistics
            features['intensity_mean'] = np.mean(gray)
            features['intensity_std'] = np.std(gray)
            features['intensity_skewness'] = self._calculate_skewness(gray)
            features['intensity_kurtosis'] = self._calculate_kurtosis(gray)
            
        except Exception as e:
            logger.error(f"Error extracting texture features: {e}")
            features = {
                'lbp_uniformity': 0.0,
                'gradient_mean': 0.0,
                'gradient_std': 0.0,
                'edge_density': 0.0,
                'intensity_mean': 0.0,
                'intensity_std': 0.0,
                'intensity_skewness': 0.0,
                'intensity_kurtosis': 0.0
            }
        
        return features
    
    def _simple_lbp(self, image: np.ndarray, radius: int = 1) -> np.ndarray:
        """Simple Local Binary Pattern implementation."""
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                binary_string = ''
                
                # 8 neighbors
                neighbors = [
                    image[i-radius, j-radius], image[i-radius, j], image[i-radius, j+radius],
                    image[i, j+radius], image[i+radius, j+radius], image[i+radius, j],
                    image[i+radius, j-radius], image[i, j-radius]
                ]
                
                for neighbor in neighbors:
                    binary_string += '1' if neighbor >= center else '0'
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp
    
    def _calculate_uniformity(self, lbp: np.ndarray) -> float:
        """Calculate uniformity of LBP patterns."""
        hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= np.sum(hist)  # Normalize
        
        # Calculate uniformity (inverse of entropy)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        uniformity = 1.0 / (1.0 + entropy)
        
        return uniformity
    
    def _calculate_skewness(self, image: np.ndarray) -> float:
        """Calculate skewness of pixel intensity distribution."""
        pixels = image.flatten().astype(float)
        mean_val = np.mean(pixels)
        std_val = np.std(pixels)
        
        if std_val == 0:
            return 0.0
        
        skewness = np.mean(((pixels - mean_val) / std_val) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, image: np.ndarray) -> float:
        """Calculate kurtosis of pixel intensity distribution."""
        pixels = image.flatten().astype(float)
        mean_val = np.mean(pixels)
        std_val = np.std(pixels)
        
        if std_val == 0:
            return 0.0
        
        kurtosis = np.mean(((pixels - mean_val) / std_val) ** 4) - 3
        return kurtosis
    
    def visualize_processing_steps(self, image: np.ndarray, 
                                 steps: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Visualize various image processing steps for debugging.
        
        Args:
            image: Input image
            steps: List of processing steps to visualize
            
        Returns:
            Dictionary of processed images for each step
        """
        if steps is None:
            steps = ['original', 'grayscale', 'denoised', 'enhanced', 'edges', 'contours']
        
        results = {}
        
        try:
            # Original
            if 'original' in steps:
                results['original'] = image.copy()
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            if 'grayscale' in steps:
                results['grayscale'] = gray
            
            # Denoised
            if 'denoised' in steps:
                denoised = self.remove_noise(gray)
                results['denoised'] = denoised
            else:
                denoised = gray
            
            # Enhanced contrast
            if 'enhanced' in steps:
                enhanced = self.enhance_contrast(denoised)
                results['enhanced'] = enhanced
            else:
                enhanced = denoised
            
            # Edge detection
            if 'edges' in steps:
                edges = self.detect_edges(enhanced)
                results['edges'] = edges
            
            # Contours
            if 'contours' in steps:
                contours = self.find_contours(enhanced if 'edges' not in steps else edges)
                contour_img = np.zeros_like(gray)
                cv2.drawContours(contour_img, contours, -1, 255, 2)
                results['contours'] = contour_img
            
            # Morphological operations
            if 'morphology' in steps:
                if 'edges' in results:
                    morph = self.morphological_operations(results['edges'], 'close')
                else:
                    edges = self.detect_edges(enhanced)
                    morph = self.morphological_operations(edges, 'close')
                results['morphology'] = morph
            
            # Skeleton
            if 'skeleton' in steps:
                if 'edges' in results:
                    skeleton = self.skeletonize(results['edges'])
                else:
                    edges = self.detect_edges(enhanced)
                    skeleton = self.skeletonize(edges)
                results['skeleton'] = skeleton
            
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
        
        return results
    
    def save_processing_results(self, results: Dict[str, np.ndarray], 
                              output_dir: str = "processing_results"):
        """
        Save processing results to files.
        
        Args:
            results: Dictionary of processed images
            output_dir: Directory to save results
        """
        import os
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for step_name, image in results.items():
                filename = f"{step_name}.png"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, image)
                logger.info(f"Saved {step_name} result to {filepath}")
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")

# Example usage and testing functions
def test_image_processor():
    """Test the ImageProcessor class with sample operations."""
    processor = ImageProcessor()
    
    # Create a test image (simple pattern)
    test_image = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(test_image, (50, 50), (150, 150), 255, 2)
    cv2.circle(test_image, (100, 100), 30, 255, 2)
    
    print("Testing ImageProcessor...")
    
    # Test preprocessing
    processed = processor.preprocess(test_image)
    print(f"Preprocessed image shape: {processed.shape}")
    
    # Test edge detection
    edges = processor.detect_edges(processed)
    print(f"Edge detection completed. Non-zero pixels: {np.count_nonzero(edges)}")
    
    # Test contour finding
    contours = processor.find_contours(edges)
    print(f"Found {len(contours)} contours")
    
    # Test symmetry analysis
    symmetry = processor.analyze_symmetry(processed)
    print(f"Symmetry scores: {symmetry}")
    
    # Test texture features
    texture = processor.extract_texture_features(processed)
    print(f"Texture features: {list(texture.keys())}")
    
    print("ImageProcessor test completed!")

if __name__ == "__main__":
    test_image_processor()