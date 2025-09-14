# Kolam AI

Kolam AI is an advanced web application for authentic Tamil Kolam pattern recognition, classification, and generation using state-of-the-art AI and deep learning. It preserves cultural heritage by analyzing, generating, and validating Kolam patterns with respect to traditional rules and regional variations.

## Features
- **Kolam Pattern Classification:** Upload Kolam images and get instant classification using a CNN-based model.
- **Pattern Generation:** Create new Kolam patterns with customizable parameters (type, grid size, symmetry, color scheme, complexity).
- **Cultural Authenticity Validation:** Ensures generated and classified patterns adhere to traditional Tamil rules and cultural constraints.
- **Interactive Web UI:** User-friendly interface for uploading, viewing results, and generating patterns.
- **API Access:** RESTful endpoints for classification and generation.
- **Comprehensive Logging:** Tracks user actions and system events for transparency and debugging.

## Project Structure
```
kolam-ai/
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── config.py             # Configuration settings
├── run.py                # Application entry point
├── static/               # Static web assets (css, js, images)
├── templates/            # Jinja2 HTML templates
├── models/               # ML models and architectures
├── utils/                # Utility modules (image processing, validation, helpers)
├── data/                 # Datasets, cultural info, processed data
├── tests/                # Test suite
├── scripts/              # Utility scripts (setup, training, evaluation)
├── docs/                 # Documentation
├── deployment/           # Deployment configs (Docker, Nginx, Heroku)
├── logs/                 # Application logs
└── backup/               # Backups
```

## Key Modules
- **models/kolam_classifier.py:** Deep learning model for Kolam type classification (MobileNet backbone).
- **models/pattern_generator.py:** Rule-based and generative model for creating new Kolam patterns.
- **utils/image_processor.py:** Preprocessing, enhancement, and feature extraction from images.
- **utils/pattern_analyzer.py:** Advanced pattern analysis (topology, complexity, symmetry).
- **utils/cultural_validator.py:** Validates patterns against traditional and regional rules.
- **utils/svg_generator.py:** Generates SVG vector graphics for patterns.
- **static/js/main.js:** Handles frontend interactivity (upload, preview, generation).

## Setup & Installation
1. **Clone the repository:**
	```sh
	git clone https://github.com/VISHWAJITH-GS/KOLAMAI.git
	cd KOLAMAI
	```
2. **Install dependencies:**
	```sh
	pip install -r requirements.txt
	```
3. **Run the application:**
	```sh
	python run.py
	```
4. **Access the app:**
	Open your browser at [http://localhost:5000](http://localhost:5000)

## Usage
- **Upload Kolam:** Use the homepage to upload a Kolam image and view classification results.
- **Generate Pattern:** Go to the generation page to create new patterns with custom parameters.
- **API:**
  - `POST /api/classify` — Classify uploaded Kolam image (multipart/form-data)
  - `POST /api/generate` — Generate Kolam pattern (JSON parameters)

## Testing
Add and run tests in the `tests/` directory to ensure reliability:
```sh
pytest tests/
```

## Contribution
Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License
This project is licensed under the MIT License.

## Acknowledgements
- Inspired by Tamil cultural heritage and Kolam artists.
- Built with Flask, TensorFlow, PyTorch, OpenCV, and more.
