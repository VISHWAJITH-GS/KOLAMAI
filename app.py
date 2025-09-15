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
# For the streamlined version, we'll use fallback implementations directly
print("Using development mode with fallback implementations")
MODELS_AVAILABLE = False

# Define helper functions that may be missing
def allowed_file(filename, allowed_extensions={'png', 'jpg', 'jpeg', 'gif', 'bmp'}):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def generate_filename(filename):
    """Generate a unique filename with timestamp"""
    import uuid
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name, ext = os.path.splitext(filename)
    return f"{name}_{timestamp}_{uuid.uuid4().hex[:8]}{ext}"

def log_activity(message):
    """Simple logging function"""
    logger = logging.getLogger(__name__)
    logger.info(message)

def cleanup_old_files(directory, max_age_days=1):
    """Remove old files from directory"""
    import time
    from datetime import datetime, timedelta
    if not os.path.exists(directory):
        return
    
    cutoff = datetime.now() - timedelta(days=max_age_days)
    cutoff_timestamp = cutoff.timestamp()
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            if os.path.getmtime(item_path) < cutoff_timestamp:
                try:
                    os.remove(item_path)
                    print(f"Removed old file: {item_path}")
                except Exception as e:
                    print(f"Error removing {item_path}: {e}")

def preprocess_image(image):
    """Preprocess image for classification"""
    # Simple preprocessing, resize to 224x224
    try:
        resized = cv2.resize(image, (224, 224))
        # Normalize
        normalized = resized / 255.0
        return normalized
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return image

def analyze_pattern(image):
    """Extract features from image"""
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

def validate_authenticity(features, classification=None):
    """Validate pattern authenticity"""
    # Simple fallback
    return 0.85

# Simplified classifier fallback
class KolamClassifier:
    def __init__(self, *args, **kwargs):
        self.CLASS_LABELS = [
            "pulli_kolam", "sikku_kolam", "kambi_kolam", 
            "padi_kolam", "rangoli", "festival_special"
        ]
        self.TAMIL_NAMES = {
            "pulli_kolam": "புள்ளி கோலம்",
            "sikku_kolam": "சிக்கு கோலம்",
            "kambi_kolam": "கம்பி கோலம்",
            "padi_kolam": "படி கோலம்", 
            "rangoli": "ரங்கோலி",
            "festival_special": "பண்டிகை சிறப்பு"
        }
    
    def predict(self, image):
        import random
        import numpy as np
        # Return mock predictions for development
        predicted_class = random.choice(self.CLASS_LABELS)
        confidence = random.uniform(0.7, 0.95)
        
        # Create random scores for all classes
        all_scores = np.random.uniform(0.1, 0.3, len(self.CLASS_LABELS))
        # Set highest score for predicted class
        idx = self.CLASS_LABELS.index(predicted_class)
        all_scores[idx] = confidence
        
        # Normalize to sum to 1
        all_scores = all_scores / all_scores.sum()
        
        # Get top 3 indices
        top_indices = np.argsort(all_scores)[::-1][:3]
        
        return {
            'predicted_class': predicted_class,
            'predicted_class_tamil': self.TAMIL_NAMES.get(predicted_class, ''),
            'confidence': float(all_scores[idx]),
            'top_3_predictions': [
                {
                    'class': self.CLASS_LABELS[i],
                    'name': self.TAMIL_NAMES.get(self.CLASS_LABELS[i], ''),
                    'confidence': float(all_scores[i])
                } for i in top_indices
            ],
            'features': list(np.random.rand(64)),  # Mock features
            'prediction_metadata': {
                'timestamp': str(datetime.now()),
                'version': '1.0-dev',
            }
        }

# Simplified generator fallback
class PatternGenerator:
    def __init__(self, *args, **kwargs):
        pass
        
    def generate(self, pattern_type=None, grid_size=None, symmetry=None, 
               color_scheme=None, complexity=None):
        # Generate a simple placeholder SVG
        import math
        width, height = 400, 400
        svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        
        # Add a simple kolam-like pattern
        cx, cy = width/2, height/2
        r = min(width, height) * 0.4
        
        # Generate a simple mandala-like pattern
        num_petals = 8
        for i in range(num_petals):
            angle = 2 * math.pi * i / num_petals
            x1 = cx + r * 0.5 * math.cos(angle)
            y1 = cy + r * 0.5 * math.sin(angle)
            x2 = cx + r * math.cos(angle)
            y2 = cy + r * math.sin(angle)
            
            svg += f'<path d="M {cx},{cy} Q {x1},{y1} {x2},{y2} Q {x1+10},{y1+10} {cx},{cy}" '
            svg += 'fill="none" stroke="black" stroke-width="2" />'
        
        # Add some dots
        for i in range(num_petals * 3):
            angle = 2 * math.pi * i / (num_petals * 3)
            x = cx + r * 0.8 * math.cos(angle)
            y = cy + r * 0.8 * math.sin(angle)
            svg += f'<circle cx="{x}" cy="{y}" r="3" fill="black" />'
        
        svg += '</svg>'
        
        return {
            'svg': svg,
            'pattern_type': pattern_type or "pulli_kolam",
            'authenticity_score': 0.85,
            'metadata': {
                'grid_size': grid_size or 9,
                'complexity': complexity or 0.7,
                'symmetry': symmetry or "ROTATIONAL_8",
            }
        }
    MODELS_AVAILABLE = False
    
    # Fallback implementations
    def allowed_file(filename, allowed_extensions={'png', 'jpg', 'jpeg', 'gif', 'bmp'}):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
    
    def preprocess_image(image):
        return image
    
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
    
    def validate_authenticity(pattern_data, classification=None):
        # Simple fallback
        return 0.85
    
    # Simplified classifier fallback
    class KolamClassifier:
        def __init__(self, *args, **kwargs):
            self.CLASS_LABELS = [
                "pulli_kolam", "sikku_kolam", "kambi_kolam", 
                "padi_kolam", "rangoli", "festival_special"
            ]
            self.TAMIL_NAMES = {
                "pulli_kolam": "புள்ளி கோலம்",
                "sikku_kolam": "சிக்கு கோலம்",
                "kambi_kolam": "கம்பி கோலம்",
                "padi_kolam": "படி கோலம்", 
                "rangoli": "ரங்கோலி",
                "festival_special": "பண்டிகை சிறப்பு"
            }
        
        def predict(self, image):
            import random
            import numpy as np
            # Return mock predictions for development
            predicted_class = random.choice(self.CLASS_LABELS)
            confidence = random.uniform(0.7, 0.95)
            
            # Create random scores for all classes
            all_scores = np.random.uniform(0.1, 0.3, len(self.CLASS_LABELS))
            # Set highest score for predicted class
            idx = self.CLASS_LABELS.index(predicted_class)
            all_scores[idx] = confidence
            
            # Normalize to sum to 1
            all_scores = all_scores / all_scores.sum()
            
            # Get top 3 indices
            top_indices = np.argsort(all_scores)[::-1][:3]
            
            return {
                'predicted_class': predicted_class,
                'predicted_class_tamil': self.TAMIL_NAMES.get(predicted_class, ''),
                'confidence': float(all_scores[idx]),
                'top_3_predictions': [
                    {
                        'class': self.CLASS_LABELS[i],
                        'name': self.TAMIL_NAMES.get(self.CLASS_LABELS[i], ''),
                        'confidence': float(all_scores[i])
                    } for i in top_indices
                ],
                'features': list(np.random.rand(64)),  # Mock features
                'prediction_metadata': {
                    'timestamp': str(datetime.now()),
                    'version': '1.0-dev',
                }
            }
    
    # Simplified generator fallback
    class PatternGenerator:
        def __init__(self, *args, **kwargs):
            pass
            
        def generate(self, pattern_type=None, grid_size=None, symmetry=None, 
                   color_scheme=None, complexity=None):
            # Generate a simple placeholder SVG
            import math
            width, height = 400, 400
            svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
            
            # Add a simple kolam-like pattern
            cx, cy = width/2, height/2
            r = min(width, height) * 0.4
            
            # Generate a simple mandala-like pattern
            num_petals = 8
            for i in range(num_petals):
                angle = 2 * math.pi * i / num_petals
                x1 = cx + r * 0.5 * math.cos(angle)
                y1 = cy + r * 0.5 * math.sin(angle)
                x2 = cx + r * math.cos(angle)
                y2 = cy + r * math.sin(angle)
                
                svg += f'<path d="M {cx},{cy} Q {x1},{y1} {x2},{y2} Q {x1+10},{y1+10} {cx},{cy}" '
                svg += 'fill="none" stroke="black" stroke-width="2" />'
            
            # Add some dots
            for i in range(num_petals * 3):
                angle = 2 * math.pi * i / (num_petals * 3)
                x = cx + r * 0.8 * math.cos(angle)
                y = cy + r * 0.8 * math.sin(angle)
                svg += f'<circle cx="{x}" cy="{y}" r="3" fill="black" />'
            
            svg += '</svg>'
            
            return {
                'svg': svg,
                'pattern_type': pattern_type or "pulli_kolam",
                'authenticity_score': 0.85,
                'metadata': {
                    'grid_size': grid_size or 9,
                    'complexity': complexity or 0.7,
                    'symmetry': symmetry or "ROTATIONAL_8",
                }
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
    # Use our MODELS_AVAILABLE flag to determine if we should use real models or fallbacks
    if MODELS_AVAILABLE:
        classifier = KolamClassifier()
        pattern_generator = PatternGenerator()
        logger.info("AI models loaded successfully")
    else:
        # Use our fallback implementations
        classifier = KolamClassifier()
        pattern_generator = PatternGenerator()
        logger.info("Using fallback AI models (development mode)")
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
            # Make sure all values are properly formatted
            logger.info(f"Features type: {type(result['features'])}")
            
            # Handle features depending on their type
            if isinstance(result['features'], list):
                try:
                    features = [float(f) if hasattr(f, 'item') else f for f in result['features']]
                except:
                    features = result['features']
            elif isinstance(result['features'], dict):
                # Ensure all dict values are proper primitive types
                features = {}
                for k, v in result['features'].items():
                    if hasattr(v, 'item'):
                        features[k] = float(v)
                    else:
                        features[k] = v
            else:
                features = result['features']
                
            # Convert confidence to percentage for display
            confidence_pct = float(result['confidence']) * 100
            
            return render_template('results.html',
                                 image_path=image_url,
                                 classification=result['classification'],
                                 confidence=confidence_pct,
                                 features=features,
                                 cultural_info=result['cultural_info'])
        else:
            flash(f'Classification failed: {result["error"]}')
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        flash('An error occurred during classification. Please try again.')
        return redirect(url_for('index'))

@app.route('/generate_pattern', methods=['GET', 'POST'])
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
                                     filename=generation_result['filename'],
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
            try:
                classification_result = classifier.predict(processed_image)
                logger.info(f"Classification result: {classification_result}")
                
                # Handle both 'predicted_class' and 'class' keys
                classification = 'sikku_kolam'  # Default fallback
                if 'predicted_class' in classification_result:
                    classification = classification_result['predicted_class']
                elif 'class' in classification_result:
                    classification = classification_result['class']
                    
                # Make sure confidence is a float
                confidence = 0.85  # Default fallback
                if 'confidence' in classification_result:
                    confidence_value = classification_result['confidence']
                    if isinstance(confidence_value, str):
                        confidence = float(confidence_value.strip('%')) / 100
                    else:
                        confidence = float(confidence_value)
                        
                logger.info(f"Final classification: {classification}, confidence: {confidence}")
            except Exception as e:
                logger.error(f"Error during classification step: {e}")
                classification = 'sikku_kolam'
                confidence = 0.85
        else:
            # Fallback classification for development
            classification = 'sikku_kolam'
            confidence = 0.85
        
        # Get cultural information
        cultural_info = cultural_db.get(classification, {
            'name': classification.replace('_', ' ').title(),
            'significance': 'Traditional Kolam pattern',
            'region': 'Tamil Nadu',
            'complexity_level': 3  # Using an integer for complexity level
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
            classification = classification_result.get('predicted_class', classification_result.get('class', 'sikku_kolam'))
            confidence = classification_result.get('confidence', 0.85)
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
        # Use the pattern generator to create pattern data
        if pattern_generator:
            pattern_data = pattern_generator.generate(
                pattern_type=params.get('pattern_type'),
                grid_size=int(params.get('grid_size', 10)),
                symmetry=params.get('symmetry', 'rotational'),
                color_scheme=params.get('color_scheme', 'traditional'),
                complexity=int(params.get('complexity', 5))
            )
        else:
            # Fallback for development
            pattern_data = {
                'svg': '<svg><circle cx="50" cy="50" r="40" stroke="black" fill="none"/></svg>',
                'pattern_type': params.get('pattern_type', 'pulli_kolam'),
                'authenticity_score': 0.8,
                'metadata': {
                    'grid_size': int(params.get('grid_size', 10)),
                    'complexity': int(params.get('complexity', 5)),
                    'symmetry': params.get('symmetry', 'rotational'),
                }
            }

        # Validate pattern authenticity
        try:
            from utils.cultural_validator import validate_pattern
            authenticity_score = validate_pattern(pattern_data, pattern_data.get('pattern_type', 'pulli_kolam'))
            if isinstance(authenticity_score, dict) and 'confidence_score' in authenticity_score:
                authenticity_score = authenticity_score['confidence_score']
        except Exception:
            authenticity_score = pattern_data.get('authenticity_score', 0.8)

        # Prepare SVG content
        svg_content = pattern_data.get('svg') or pattern_data.get('svg_content')
        parameters = {
            'pattern_type': pattern_data.get('pattern_type', params.get('pattern_type', 'pulli_kolam')),
            'grid_size': pattern_data.get('metadata', {}).get('grid_size', params.get('grid_size', 10)),
            'symmetry': pattern_data.get('metadata', {}).get('symmetry', params.get('symmetry', 'rotational')),
            'complexity': pattern_data.get('metadata', {}).get('complexity', params.get('complexity', 5)),
            'color_scheme': params.get('color_scheme', 'traditional')
        }

        # Save SVG file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"pattern_{uuid.uuid4().hex[:8]}_{timestamp}.svg"
        svg_folder = os.path.join(GENERATED_FOLDER, 'svg')
        os.makedirs(svg_folder, exist_ok=True)
        file_path = os.path.join(svg_folder, filename)
        with open(file_path, 'w') as f:
            f.write(svg_content)

        return {
            'success': True,
            'svg_content': svg_content,
            'parameters': parameters,
            'authenticity_score': authenticity_score,
            'file_path': file_path,
            'filename': filename
        }
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