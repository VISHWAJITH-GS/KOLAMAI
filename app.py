import os
import json
import uuid
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import cv2
import numpy as np
from PIL import Image
import logging

# Import custom modules (these would be your actual model files)
try:
    from models.kolam_classifier import KolamClassifier
    from models.pattern_generator import PatternGenerator
    from utils.image_processor import preprocess_image
    from utils.pattern_analyzer import analyze_pattern
    from utils.cultural_validator import validate_authenticity
    from utils.helpers import allowed_file, generate_filename, cleanup_old_files, log_activity
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some features may not work until all modules are implemented.")
    # Fallback allowed_file implementation
    def allowed_file(filename, allowed_extensions={'png', 'jpg', 'jpeg', 'gif', 'bmp'}):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
    # Fallback preprocess_image implementation
    def preprocess_image(image):
        return image
    # Fallback analyze_pattern implementation
    def analyze_pattern(image):
        # Return default features for development
        return {
            'num_components': 1,
            'num_junctions': 0,
            'num_loops': 0,
            'structural_complexity': 0.5,
            'overall': 0.5,
            'dot_density': 0.1,
            'line_curve_ratio': 0.5
        }

# Flask app initialization
app = Flask(__name__)
app.secret_key = 'kolam-ai-secret-key-change-in-production'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
GENERATED_FOLDER = 'static/generated'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
CLEANUP_HOURS = 24

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kolam_ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize AI models (with error handling for development)
try:
    classifier = KolamClassifier()
    pattern_generator = PatternGenerator()
    logger.info("AI models loaded successfully")
except Exception as e:
    logger.warning(f"Could not load AI models: {e}")
    logger.warning("KolamClassifier and/or PatternGenerator are not available. ML features will be disabled.")
    classifier = None
    pattern_generator = None

# Load cultural database
try:
    with open('data/cultural_database.json', 'r') as f:
        cultural_db = json.load(f)
except FileNotFoundError:
    logger.warning("Cultural database not found, using empty database")
    cultural_db = {}

@app.route('/')
def index():
    """Homepage with upload interface and recent patterns"""
    from datetime import datetime
    current_year = datetime.now().year
    try:
        # Get recent classifications for gallery
        recent_patterns = get_recent_patterns(limit=6)
        # Get statistics
        statistics = {
            'total_classifications': len(get_all_classifications()),
            'patterns_generated': len(get_generated_patterns()),
            'cultural_authenticity': calculate_average_authenticity()
        }
        return render_template('index.html', 
                             recent_patterns=recent_patterns,
                             statistics=statistics,
                             current_year=current_year)
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        return render_template('index.html', 
                             recent_patterns=[],
                             statistics={},
                             current_year=current_year)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and initial processing"""
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                flash('No file selected')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No file selected')
                return redirect(request.url)
            
            # Validate file
            if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
                flash('Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP files.')
                return redirect(request.url)
            
            # Check file size
            if len(file.read()) > MAX_FILE_SIZE:
                flash('File too large. Maximum size is 5MB.')
                return redirect(request.url)
            file.seek(0)  # Reset file pointer
            
            # Generate secure filename
            filename = generate_filename(secure_filename(file.filename))
            
            # Create date-based subfolder
            date_folder = datetime.now().strftime('%Y-%m-%d')
            upload_path = os.path.join(UPLOAD_FOLDER, date_folder)
            os.makedirs(upload_path, exist_ok=True)
            
            # Save file
            file_path = os.path.join(upload_path, filename)
            file.save(file_path)
            
            # Store file info in session
            session['uploaded_file'] = file_path
            session['upload_time'] = datetime.now().isoformat()
            
            # Log activity
            log_activity(f"File uploaded: {filename}")
            
            # Redirect to classification
            flash('File uploaded successfully!')
            return redirect(url_for('classify_pattern'))
            
        except Exception as e:
            logger.error(f"Error in file upload: {e}")
            flash('An error occurred during file upload. Please try again.')
            return redirect(request.url)
    
    return render_template('index.html')

@app.route('/classify')
def classify_pattern():
    """Classify uploaded Kolam pattern"""
    try:
        # Check if file was uploaded
        if 'uploaded_file' not in session:
            flash('Please upload an image first.')
            return redirect(url_for('index'))
        
        file_path = session['uploaded_file']
        # Check if file exists
        if not os.path.exists(file_path):
            flash('Uploaded file not found. Please upload again.')
            return redirect(url_for('index'))
        # Process image
        result = process_image(file_path)
        if result['success']:
            # Store results in session
            session['classification_result'] = result
            # Convert file_path to URL path for static serving
            if file_path.startswith('static/'):
                image_url = '/' + file_path.replace('\\', '/').replace('static/', 'static/')
            else:
                image_url = file_path
            return render_template('results.html',
                                 image_path=image_url,
                                 classification=result['classification'],
                                 confidence=result['confidence'],
                                 features=result['features'],
                                 cultural_info=result['cultural_info'])
        else:
            flash(f'Classification failed: {result["error"]}')
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        flash('An error occurred during classification. Please try again.')
        return redirect(url_for('index'))

@app.route('/generate', methods=['GET', 'POST'])
def generate_pattern():
    """Generate new Kolam patterns"""
    if request.method == 'POST':
        try:
            # Get parameters from form
            pattern_type = request.form.get('pattern_type', 'sikku_kolam')
            grid_size = int(request.form.get('grid_size', 10))
            symmetry = request.form.get('symmetry', 'rotational')
            color_scheme = request.form.get('color_scheme', 'traditional')
            complexity = int(request.form.get('complexity', 5))
            
            # Validate parameters
            if grid_size < 5 or grid_size > 20:
                flash('Grid size must be between 5 and 20.')
                return redirect(request.url)
            
            # Generate pattern
            generation_result = generate_kolam_pattern(
                pattern_type=pattern_type,
                grid_size=grid_size,
                symmetry=symmetry,
                color_scheme=color_scheme,
                complexity=complexity
            )
            
            if generation_result['success']:
                # Store generation info in session
                session['generated_pattern'] = generation_result
                
                return render_template('generate.html',
                                     generated_svg=generation_result['svg_content'],
                                     parameters=generation_result['parameters'],
                                     file_path=generation_result['file_path'],
                                     authenticity_score=generation_result['authenticity_score'])
            else:
                flash(f'Pattern generation failed: {generation_result["error"]}')
                return redirect(request.url)
                
        except Exception as e:
            logger.error(f"Error in pattern generation: {e}")
            flash('An error occurred during pattern generation. Please try again.')
            return redirect(request.url)
    
    # Pass default values to prevent template errors on GET
    return render_template('generate.html',
        generated_svg=None,
        parameters=None,
        file_path=None,
        authenticity_score=None,
        confidence=None,
        features={}
    )

@app.route('/api/classify', methods=['POST'])
def api_classify():
    """API endpoint for pattern classification"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Process image directly from memory
        image_data = file.read()
        result = process_image_data(image_data)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in API classification: {e}")
        return jsonify({'error': 'Classification failed'}), 500

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API endpoint for pattern generation"""
    try:
        data = request.get_json()
        
        # Validate required parameters
        required_params = ['pattern_type', 'grid_size']
        for param in required_params:
            if param not in data:
                return jsonify({'error': f'Missing parameter: {param}'}), 400
        
        # Generate pattern
        result = generate_kolam_pattern(**data)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in API generation: {e}")
        return jsonify({'error': 'Generation failed'}), 500

def process_image(file_path):
    """Process uploaded image and classify pattern"""
    try:
        # Load and preprocess image
        image = cv2.imread(file_path)
        if image is None:
            return {'success': False, 'error': 'Could not load image'}
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Extract features
        features = analyze_pattern(processed_image)
        
        # Classify pattern
        if classifier:
            classification_result = classifier.predict(processed_image)
            classification = classification_result['class']
            confidence = classification_result['confidence']
        else:
            # Fallback classification for development
            classification = 'sikku_kolam'
            confidence = 0.85
        
        # Get cultural information
        cultural_info = cultural_db.get(classification, {
            'name': classification.replace('_', ' ').title(),
            'significance': 'Traditional Kolam pattern',
            'region': 'Tamil Nadu',
            'complexity_level': 'Medium'
        })
        
        # Validate authenticity
        authenticity_score = validate_authenticity(features, classification)
        
        return {
            'success': True,
            'classification': classification,
            'confidence': confidence,
            'features': features,
            'cultural_info': cultural_info,
            'authenticity_score': authenticity_score
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {'success': False, 'error': str(e)}

def process_image_data(image_data):
    """Process image data from API request"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {'success': False, 'error': 'Could not decode image'}
        
        # Use same processing logic as file upload
        processed_image = preprocess_image(image)
        features = analyze_pattern(processed_image)
        
        if classifier:
            classification_result = classifier.predict(processed_image)
            classification = classification_result['class']
            confidence = classification_result['confidence']
        else:
            classification = 'sikku_kolam'
            confidence = 0.85
        
        cultural_info = cultural_db.get(classification, {})
        authenticity_score = validate_authenticity(features, classification)
        
        return {
            'success': True,
            'classification': classification,
            'confidence': confidence,
            'features': features,
            'cultural_info': cultural_info,
            'authenticity_score': authenticity_score
        }
        
    except Exception as e:
        logger.error(f"Error processing image data: {e}")
        return {'success': False, 'error': str(e)}

def generate_kolam_pattern(**params):
    """Generate Kolam pattern with given parameters"""
    try:
        if pattern_generator:
            # Use actual pattern generator
            result = pattern_generator.generate(**params)
        else:
            # Fallback for development
            result = {
                'svg_content': '<svg><circle cx="50" cy="50" r="40" stroke="black" fill="none"/></svg>',
                'parameters': params,
                'authenticity_score': 0.8
            }
        
        # Generate filename and save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"pattern_{uuid.uuid4().hex[:8]}_{timestamp}.svg"
        file_path = os.path.join(GENERATED_FOLDER, filename)
        
        # Save SVG file
        with open(file_path, 'w') as f:
            f.write(result['svg_content'])
        
        result.update({
            'success': True,
            'file_path': file_path,
            'filename': filename
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating pattern: {e}")
        return {'success': False, 'error': str(e)}

def get_recent_patterns(limit=6):
    """Get recent classification results"""
    try:
        # This would typically query a database
        # For now, return mock data
        return [
            {'image': 'sample1.jpg', 'type': 'Sikku Kolam', 'confidence': 0.92},
            {'image': 'sample2.jpg', 'type': 'Pulli Kolam', 'confidence': 0.87},
            {'image': 'sample3.jpg', 'type': 'Festival Pattern', 'confidence': 0.95},
        ][:limit]
    except Exception as e:
        logger.error(f"Error getting recent patterns: {e}")
        return []

def get_all_classifications():
    """Get all classification records"""
    # Mock implementation
    return []

def get_generated_patterns():
    """Get all generated patterns"""
    try:
        patterns = []
        if os.path.exists(GENERATED_FOLDER):
            for filename in os.listdir(GENERATED_FOLDER):
                if filename.endswith('.svg'):
                    patterns.append(filename)
        return patterns
    except Exception as e:
        logger.error(f"Error getting generated patterns: {e}")
        return []

def calculate_average_authenticity():
    """Calculate average authenticity score"""
    # Mock implementation
    return 0.85

@app.before_request
def cleanup_old_files():
    """Clean up old uploaded files"""
    try:
        cleanup_time = datetime.now() - timedelta(hours=CLEANUP_HOURS)
        # Implementation would clean files older than cleanup_time
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Create sample data directory structure
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('utils', exist_ok=True)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)