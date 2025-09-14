"""
Kolam AI - Flask Backend Application
Main Flask application handling web routes, file uploads, and API endpoints
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
from flask import send_from_directory, abort
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import logging
from functools import wraps
import time

# Import custom modules
from utils.image_processor import ImageProcessor
from utils.pattern_analyzer import PatternAnalyzer
from utils.cultural_validator import CulturalValidator
from utils.helpers import allowed_file, generate_filename, cleanup_old_files, format_file_size
from models.kolam_classifier import KolamClassifier
from models.pattern_generator import PatternGenerator
from models.feature_extractor import FeatureExtractor

# Initialize Flask application
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'kolam-ai-secret-key-2024')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size

# File upload configuration
UPLOAD_FOLDER = 'static/uploads'
GENERATED_FOLDER = 'static/generated'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}
TEMP_FILE_LIFETIME = 3600  # 1 hour in seconds

# Ensure upload directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize ML models and processors (lazy loading for performance)
image_processor = None
pattern_analyzer = None
cultural_validator = None
kolam_classifier = None
pattern_generator = None
feature_extractor = None

def init_models():
    """Initialize ML models and processors lazily"""
    global image_processor, pattern_analyzer, cultural_validator
    global kolam_classifier, pattern_generator, feature_extractor
    
    try:
        logger.info("Initializing models...")
        image_processor = ImageProcessor()
        pattern_analyzer = PatternAnalyzer()
        cultural_validator = CulturalValidator()
        kolam_classifier = KolamClassifier()
        pattern_generator = PatternGenerator()
        feature_extractor = FeatureExtractor()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        # For demo purposes, continue without models
        pass

# Decorator for timing requests
def time_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{f.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return decorated_function

# Error handlers
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        'error': 'File too large',
        'message': 'File size must be less than 5MB',
        'max_size': '5MB'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong. Please try again.'
    }), 500

@app.errorhandler(404)
def not_found(error):
    if request.is_json:
        return jsonify({
            'error': 'Not found',
            'message': 'The requested resource was not found'
        }), 404
    return render_template('404.html'), 404

# Routes

@app.route('/')
@time_request
def index():
    """Homepage route"""
    try:
        # Get recent patterns for homepage display
        recent_patterns = get_recent_patterns()
        
        # Get app statistics
        stats = {
            'total_patterns': get_total_patterns_count(),
            'classifications_today': get_today_classifications(),
            'patterns_generated': get_generated_count(),
            'cultural_database_size': get_cultural_db_size()
        }
        
        return render_template('index.html', 
                             recent_patterns=recent_patterns,
                             stats=stats)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return render_template('index.html', recent_patterns=[], stats={})

@app.route('/upload')
def upload_page():
    """Upload page route"""
    return render_template('upload.html')

@app.route('/generate')
def generate_page():
    """Pattern generation page route"""
    return render_template('generate.html')

@app.route('/gallery')
def gallery():
    """Gallery page route"""
    try:
        # Get gallery patterns with pagination
        page = request.args.get('page', 1, type=int)
        category = request.args.get('category', 'all')
        patterns = get_gallery_patterns(page, category)
        
        return render_template('gallery.html', patterns=patterns)
    except Exception as e:
        logger.error(f"Error in gallery route: {str(e)}")
        return render_template('gallery.html', patterns=[])

@app.route('/about')
def about():
    """About page route"""
    return render_template('about.html')

# API Routes

@app.route('/api/upload', methods=['POST'])
@time_request
def upload_file():
    """Handle file upload and return file info"""
    try:
        # Initialize models if not already done
        if image_processor is None:
            init_models()
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file
        if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, WEBP, GIF'
            }), 400
        
        # Generate secure filename
        filename = generate_filename(secure_filename(file.filename))
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save file
        file.save(filepath)
        
        # Get file info
        file_stats = os.stat(filepath)
        file_info = {
            'filename': filename,
            'original_name': file.filename,
            'size': file_stats.st_size,
            'size_formatted': format_file_size(file_stats.st_size),
            'upload_time': datetime.now().isoformat(),
            'file_path': filepath,
            'url': url_for('static', filename=f'uploads/{filename}')
        }
        
        # Basic image validation
        try:
            if image_processor:
                image_info = image_processor.get_image_info(filepath)
                file_info.update(image_info)
        except Exception as e:
            logger.warning(f"Could not extract image info: {str(e)}")
        
        # Store file info in session
        session['uploaded_file'] = file_info
        
        # Schedule cleanup
        cleanup_old_files(UPLOAD_FOLDER, TEMP_FILE_LIFETIME)
        
        logger.info(f"File uploaded successfully: {filename}")
        
        return jsonify({
            'success': True,
            'file_info': file_info,
            'message': 'File uploaded successfully'
        })
        
    except RequestEntityTooLarge:
        return jsonify({
            'success': False,
            'error': 'File too large (max 5MB)'
        }), 413
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Upload failed. Please try again.'
        }), 500

@app.route('/api/classify', methods=['POST'])
@time_request
def classify_pattern():
    """Classify uploaded Kolam pattern"""
    try:
        # Initialize models if needed
        if kolam_classifier is None:
            init_models()
        
        # Get file info from request or session
        file_info = request.get_json() if request.is_json else session.get('uploaded_file')
        
        if not file_info:
            return jsonify({
                'success': False,
                'error': 'No file to classify'
            }), 400
        
        filepath = file_info.get('file_path')
        if not filepath or not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        # Get analysis options
        options = request.form.getlist('options[]') if request.form else ['symmetry', 'cultural']
        
        # Process image
        processed_results = process_image(filepath, options)
        
        # Store results in session
        session['classification_results'] = processed_results
        session['analysis_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Pattern classified: {processed_results['classification']['type']}")
        
        return jsonify({
            'success': True,
            'results': processed_results,
            'message': 'Classification completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Classification failed. Please try again.',
            'details': str(e) if app.debug else None
        }), 500

@app.route('/api/generate', methods=['POST'])
@time_request
def generate_pattern():
    """Generate new Kolam pattern based on parameters"""
    try:
        # Initialize models if needed
        if pattern_generator is None:
            init_models()
        
        # Get generation parameters
        data = request.get_json() or {}
        
        params = {
            'pattern_type': data.get('pattern_type', 'pulli_kolam'),
            'grid_size': data.get('grid_size', 7),
            'symmetry': data.get('symmetry', 4),
            'complexity': data.get('complexity', 'medium'),
            'color_scheme': data.get('color_scheme', 'traditional'),
            'cultural_authenticity': data.get('cultural_authenticity', 'high')
        }
        
        # Validate parameters
        validation_result = validate_generation_params(params)
        if not validation_result['valid']:
            return jsonify({
                'success': False,
                'error': 'Invalid parameters',
                'details': validation_result['errors']
            }), 400
        
        # Generate pattern
        generation_result = generate_kolam_pattern(params)
        
        if not generation_result['success']:
            return jsonify({
                'success': False,
                'error': 'Pattern generation failed',
                'details': generation_result.get('error')
            }), 500
        
        # Store generation results
        session['generated_pattern'] = generation_result
        session['generation_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Pattern generated: {params['pattern_type']}")
        
        return jsonify({
            'success': True,
            'pattern': generation_result,
            'message': 'Pattern generated successfully'
        })
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Pattern generation failed. Please try again.',
            'details': str(e) if app.debug else None
        }), 500

@app.route('/results')
def results():
    """Display classification results"""
    try:
        results = session.get('classification_results')
        file_info = session.get('uploaded_file')
        
        if not results or not file_info:
            flash('No results to display. Please upload and analyze an image first.', 'warning')
            return redirect(url_for('upload_page'))
        
        return render_template('results.html', 
                             results=results, 
                             file_info=file_info)
    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}")
        flash('Error displaying results. Please try again.', 'error')
        return redirect(url_for('upload_page'))

@app.route('/api/sample/<pattern_type>')
def get_sample_pattern(pattern_type):
    """Get sample pattern for quick demo"""
    try:
        sample_path = f'data/sample_patterns/{pattern_type}/'
        if os.path.exists(sample_path):
            files = [f for f in os.listdir(sample_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                sample_file = files[0]  # Get first available sample
                return send_from_directory(sample_path, sample_file)
        
        abort(404)
    except Exception as e:
        logger.error(f"Sample pattern error: {str(e)}")
        abort(404)

# Helper Functions

def process_image(filepath, options):
    """Process image through the complete analysis pipeline"""
    try:
        results = {
            'file_path': filepath,
            'processing_time': time.time(),
            'options': options
        }
        
        # Step 1: Image preprocessing
        if image_processor:
            processed_image = image_processor.preprocess_image(filepath)
            results['processed_image'] = processed_image
        
        # Step 2: Feature extraction
        if feature_extractor:
            features = feature_extractor.extract_features(filepath)
            results['features'] = features
        
        # Step 3: Pattern classification
        if kolam_classifier:
            classification = kolam_classifier.classify_pattern(filepath)
            results['classification'] = classification
        else:
            # Fallback classification for demo
            results['classification'] = get_demo_classification(filepath)
        
        # Step 4: Pattern analysis (if requested)
        if 'symmetry' in options and pattern_analyzer:
            symmetry_analysis = pattern_analyzer.analyze_symmetry(filepath)
            results['symmetry'] = symmetry_analysis
        
        # Step 5: Cultural context (if requested)
        if 'cultural' in options and cultural_validator:
            cultural_info = cultural_validator.get_cultural_context(results['classification']['type'])
            results['cultural_info'] = cultural_info
        else:
            results['cultural_info'] = get_demo_cultural_info()
        
        # Step 6: Similar patterns (if requested)
        if 'similar' in options:
            similar_patterns = find_similar_patterns(results['classification']['type'])
            results['similar_patterns'] = similar_patterns
        
        results['processing_time'] = time.time() - results['processing_time']
        return results
        
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise

def generate_kolam_pattern(params):
    """Generate Kolam pattern based on parameters"""
    try:
        if pattern_generator:
            result = pattern_generator.generate_pattern(params)
        else:
            # Demo generation
            result = get_demo_generation(params)
        
        # Save generated pattern
        if result['success']:
            pattern_id = str(uuid.uuid4())[:8]
            filename = f"{params['pattern_type']}_{pattern_id}.svg"
            filepath = os.path.join(GENERATED_FOLDER, filename)
            
            # Save SVG file
            with open(filepath, 'w') as f:
                f.write(result['svg_content'])
            
            result['file_path'] = filepath
            result['filename'] = filename
            result['url'] = url_for('static', filename=f'generated/{filename}')
        
        return result
        
    except Exception as e:
        logger.error(f"Pattern generation error: {str(e)}")
        return {'success': False, 'error': str(e)}

def validate_generation_params(params):
    """Validate pattern generation parameters"""
    errors = []
    
    # Validate pattern type
    valid_types = ['pulli_kolam', 'sikku_kolam', 'rangoli', 'festival']
    if params['pattern_type'] not in valid_types:
        errors.append(f"Invalid pattern type. Must be one of: {', '.join(valid_types)}")
    
    # Validate grid size
    if not (3 <= params['grid_size'] <= 15):
        errors.append("Grid size must be between 3 and 15")
    
    # Validate symmetry
    if params['symmetry'] not in [2, 4, 6, 8, 12, 16]:
        errors.append("Symmetry must be 2, 4, 6, 8, 12, or 16")
    
    # Validate complexity
    valid_complexity = ['simple', 'medium', 'complex']
    if params['complexity'] not in valid_complexity:
        errors.append(f"Complexity must be one of: {', '.join(valid_complexity)}")
    
    return {'valid': len(errors) == 0, 'errors': errors}

# Demo functions (for when ML models aren't available)
def get_demo_classification(filepath):
    """Demo classification for testing without ML models"""
    return {
        'type': 'pulli_kolam',
        'name': 'Pulli Kolam',
        'description': 'Traditional dot-connected pattern',
        'confidence': 0.92,
        'alternatives': [
            {'name': 'Sikku Kolam', 'confidence': 0.06},
            {'name': 'Rangoli', 'confidence': 0.02}
        ]
    }

def get_demo_cultural_info():
    """Demo cultural information"""
    return {
        'title': 'Traditional Pulli Kolam',
        'tamil_name': 'புள்ளி கோலம்',
        'description': 'A sacred geometric pattern drawn with rice flour, representing prosperity and welcoming positive energy.',
        'significance': 'Traditionally drawn at dawn by women to welcome guests and invite prosperity.',
        'region': 'Tamil Nadu, South India',
        'season': 'Daily practice, special variations during festivals'
    }

def get_demo_generation(params):
    """Demo pattern generation"""
    svg_content = f'''
    <svg width="400" height="400" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <pattern id="dots" patternUnits="userSpaceOnUse" width="40" height="40">
                <circle cx="20" cy="20" r="2" fill="#DC143C"/>
            </pattern>
        </defs>
        <rect width="400" height="400" fill="url(#dots)"/>
        <path d="M 100 100 L 300 100 L 300 300 L 100 300 Z" 
              fill="none" stroke="#DC143C" stroke-width="3"/>
    </svg>
    '''
    
    return {
        'success': True,
        'svg_content': svg_content,
        'parameters': params,
        'authenticity_score': 0.85,
        'cultural_validation': 'Traditional rules followed'
    }

# Utility functions for homepage
def get_recent_patterns():
    """Get recent patterns for homepage display"""
    # Demo data
    return [
        {'id': 1, 'type': 'pulli_kolam', 'image': 'sample1.jpg'},
        {'id': 2, 'type': 'sikku_kolam', 'image': 'sample2.jpg'},
    ]

def get_total_patterns_count():
    return 150

def get_today_classifications():
    return 23

def get_generated_count():
    return 89

def get_cultural_db_size():
    return 500

def get_gallery_patterns(page, category):
    """Get gallery patterns with pagination"""
    # Demo implementation
    return []

def find_similar_patterns(pattern_type):
    """Find similar patterns in database"""
    return []

# Context processor for template variables
@app.context_processor
def inject_globals():
    return {
        'current_year': datetime.now().year,
        'app_version': '1.0.0',
        'debug_mode': app.debug
    }

# Initialize models on startup (comment out for faster startup during development)
# init_models()

if __name__ == '__main__':
    # Development server
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )