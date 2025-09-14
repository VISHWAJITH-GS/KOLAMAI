// Kolam AI - Main JavaScript File
// Interactive functionality for file upload, classification, and pattern generation

// Global variables
let currentImage = null;
let classificationResult = null;
let generatedPattern = null;
let isUploading = false;
let isClassifying = false;
let isGenerating = false;

// DOM elements
const uploadArea = document.querySelector('.upload-area');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const resultsContainer = document.getElementById('results-container');
const loadingOverlay = document.getElementById('loading-overlay');
const generateForm = document.getElementById('generate-form');
const patternPreview = document.getElementById('pattern-preview');

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeTooltips();
    initializeAnimations();
    setupFormValidation();
    console.log('Kolam AI initialized successfully');
});

// Initialize all event listeners
function initializeEventListeners() {
    // File upload events
    if (uploadArea) {
        uploadArea.addEventListener('click', () => fileInput?.click());
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
    }
    
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Classification button
    const classifyBtn = document.getElementById('classify-btn');
    if (classifyBtn) {
        classifyBtn.addEventListener('click', classifyImage);
    }
    
    // Pattern generation form
    if (generateForm) {
        generateForm.addEventListener('submit', handlePatternGeneration);
        
        // Real-time parameter updates
        const sliders = generateForm.querySelectorAll('input[type="range"]');
        sliders.forEach(slider => {
            slider.addEventListener('input', updateSliderValue);
            slider.addEventListener('change', previewPatternChanges);
        });
        
        const selects = generateForm.querySelectorAll('select');
        selects.forEach(select => {
            select.addEventListener('change', previewPatternChanges);
        });
    }
    
    // Download buttons
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('download-btn')) {
            handleDownload(e.target.dataset.format);
        }
    });
    
    // Modal events
    const modals = document.querySelectorAll('[data-toggle="modal"]');
    modals.forEach(modal => {
        modal.addEventListener('click', showModal);
    });
    
    // Gallery events
    const galleryItems = document.querySelectorAll('.gallery-item');
    galleryItems.forEach(item => {
        item.addEventListener('click', showGalleryModal);
    });
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('drag-over');
    
    // Add visual feedback
    const dragText = uploadArea.querySelector('.upload-text');
    if (dragText) {
        dragText.textContent = 'Drop your Kolam image here!';
    }
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');
    
    // Reset text
    const dragText = uploadArea.querySelector('.upload-text');
    if (dragText) {
        dragText.textContent = 'Click to upload or drag and drop';
    }
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

// Handle file selection from input
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

// Process selected file
function processFile(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        showAlert('Please select a valid image file (JPEG, PNG, GIF, or BMP)', 'error');
        return;
    }
    
    // Validate file size (5MB limit)
    const maxSize = 5 * 1024 * 1024;
    if (file.size > maxSize) {
        showAlert('File size must be less than 5MB', 'error');
        return;
    }
    
    currentImage = file;
    previewFile(file);
    enableClassification();
}

// Preview uploaded file
function previewFile(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        // Create preview element
        const previewHtml = `
            <div class="image-preview">
                <img src="${e.target.result}" alt="Uploaded Kolam" class="preview-image">
                <div class="image-info">
                    <p><strong>File:</strong> ${file.name}</p>
                    <p><strong>Size:</strong> ${formatFileSize(file.size)}</p>
                    <p><strong>Type:</strong> ${file.type}</p>
                </div>
                <button type="button" class="btn btn-sm btn-outline-danger remove-image">
                    <i class="fas fa-times"></i> Remove
                </button>
            </div>
        `;
        
        if (previewContainer) {
            previewContainer.innerHTML = previewHtml;
            previewContainer.style.display = 'block';
            
            // Add remove functionality
            const removeBtn = previewContainer.querySelector('.remove-image');
            removeBtn.addEventListener('click', clearPreview);
        }
        
        // Update upload area
        const uploadText = uploadArea?.querySelector('.upload-text');
        if (uploadText) {
            uploadText.textContent = 'Image uploaded successfully!';
        }
        
        // Add success animation
        uploadArea?.classList.add('upload-success');
        setTimeout(() => {
            uploadArea?.classList.remove('upload-success');
        }, 2000);
    };
    
    reader.readAsDataURL(file);
}

// Clear image preview
function clearPreview() {
    currentImage = null;
    classificationResult = null;
    
    if (previewContainer) {
        previewContainer.innerHTML = '';
        previewContainer.style.display = 'none';
    }
    
    if (resultsContainer) {
        resultsContainer.innerHTML = '';
        resultsContainer.style.display = 'none';
    }
    
    // Reset upload area
    const uploadText = uploadArea?.querySelector('.upload-text');
    if (uploadText) {
        uploadText.textContent = 'Click to upload or drag and drop';
    }
    
    // Reset file input
    if (fileInput) {
        fileInput.value = '';
    }
    
    disableClassification();
}

// Enable classification button
function enableClassification() {
    const classifyBtn = document.getElementById('classify-btn');
    if (classifyBtn) {
        classifyBtn.disabled = false;
        classifyBtn.classList.remove('btn-secondary');
        classifyBtn.classList.add('btn-primary-custom');
    }
}

// Disable classification button
function disableClassification() {
    const classifyBtn = document.getElementById('classify-btn');
    if (classifyBtn) {
        classifyBtn.disabled = true;
        classifyBtn.classList.remove('btn-primary-custom');
        classifyBtn.classList.add('btn-secondary');
    }
}

// Upload and classify image
async function uploadImage() {
    if (!currentImage || isUploading) return;
    
    isUploading = true;
    showLoading('Uploading and processing your Kolam image...');
    
    try {
        const formData = new FormData();
        formData.append('file', currentImage);
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            // Redirect to classification page
            window.location.href = '/classify';
        } else {
            const error = await response.text();
            throw new Error(error);
        }
        
    } catch (error) {
        console.error('Upload error:', error);
        showAlert('Failed to upload image. Please try again.', 'error');
    } finally {
        isUploading = false;
        hideLoading();
    }
}

// Classify image using API
async function classifyImage() {
    if (!currentImage || isClassifying) return;
    
    isClassifying = true;
    showLoading('Analyzing your Kolam pattern...');
    
    try {
        const formData = new FormData();
        formData.append('file', currentImage);
        
        const response = await fetch('/api/classify', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            classificationResult = result;
            showResults(result);
        } else {
            throw new Error(result.error || 'Classification failed');
        }
        
    } catch (error) {
        console.error('Classification error:', error);
        showAlert('Failed to classify image. Please try again.', 'error');
    } finally {
        isClassifying = false;
        hideLoading();
    }
}

// Display classification results
function showResults(result) {
    const resultsHtml = `
        <div class="result-card">
            <div class="result-header">
                <h3 class="result-title">${result.classification.replace('_', ' ').toUpperCase()}</h3>
                <div class="confidence-badge">
                    Confidence: ${(result.confidence * 100).toFixed(1)}%
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <h5>Pattern Features:</h5>
                    <ul class="feature-list">
                        ${Object.entries(result.features || {}).map(([key, value]) => `
                            <li class="feature-item">
                                <span class="feature-name">${key.replace('_', ' ')}:</span>
                                <span class="feature-value">${value}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
                
                <div class="col-md-6">
                    <div class="cultural-panel">
                        <h5 class="cultural-title">Cultural Significance</h5>
                        <p><strong>Name:</strong> ${result.cultural_info?.name || 'Traditional Kolam'}</p>
                        <p><strong>Region:</strong> ${result.cultural_info?.region || 'Tamil Nadu'}</p>
                        <p><strong>Significance:</strong> ${result.cultural_info?.significance || 'Sacred geometric pattern'}</p>
                        <p><strong>Complexity:</strong> ${result.cultural_info?.complexity_level || 'Medium'}</p>
                    </div>
                    
                    <div class="authenticity-score">
                        <h6>Authenticity Score:</h6>
                        <div class="progress">
                            <div class="progress-bar" style="width: ${(result.authenticity_score * 100).toFixed(0)}%">
                                ${(result.authenticity_score * 100).toFixed(0)}%
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="result-actions mt-3">
                <button class="btn btn-primary-custom" onclick="generateSimilarPattern()">
                    Generate Similar Pattern
                </button>
                <button class="btn btn-outline-custom" onclick="shareResult()">
                    Share Result
                </button>
            </div>
        </div>
    `;
    
    if (resultsContainer) {
        resultsContainer.innerHTML = resultsHtml;
        resultsContainer.style.display = 'block';
        
        // Animate results appearance
        resultsContainer.style.opacity = '0';
        resultsContainer.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            resultsContainer.style.transition = 'all 0.5s ease';
            resultsContainer.style.opacity = '1';
            resultsContainer.style.transform = 'translateY(0)';
        }, 100);
    }
    
    // Scroll to results
    resultsContainer?.scrollIntoView({ behavior: 'smooth' });
}

// Handle pattern generation form
async function handlePatternGeneration(e) {
    e.preventDefault();
    
    if (isGenerating) return;
    
    isGenerating = true;
    showLoading('Generating your authentic Kolam pattern...');
    
    try {
        const formData = new FormData(generateForm);
        const parameters = Object.fromEntries(formData.entries());
        
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(parameters)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            generatedPattern = result;
            displayGeneratedPattern(result);
        } else {
            throw new Error(result.error || 'Pattern generation failed');
        }
        
    } catch (error) {
        console.error('Generation error:', error);
        showAlert('Failed to generate pattern. Please try again.', 'error');
    } finally {
        isGenerating = false;
        hideLoading();
    }
}

// Display generated pattern
function displayGeneratedPattern(result) {
    const patternHtml = `
        <div class="generated-pattern">
            <div class="pattern-display">
                ${result.svg_content}
            </div>
            
            <div class="pattern-info">
                <h4>Generated Kolam Pattern</h4>
                <div class="pattern-params">
                    <p><strong>Type:</strong> ${result.parameters.pattern_type.replace('_', ' ')}</p>
                    <p><strong>Grid Size:</strong> ${result.parameters.grid_size}x${result.parameters.grid_size}</p>
                    <p><strong>Symmetry:</strong> ${result.parameters.symmetry}</p>
                    <p><strong>Authenticity:</strong> ${(result.authenticity_score * 100).toFixed(0)}%</p>
                </div>
                
                <div class="download-options">
                    <button class="btn btn-primary-custom download-btn" data-format="svg">
                        <i class="fas fa-download"></i> Download SVG
                    </button>
                    <button class="btn btn-outline-custom download-btn" data-format="png">
                        <i class="fas fa-download"></i> Download PNG
                    </button>
                </div>
            </div>
        </div>
    `;
    
    if (patternPreview) {
        patternPreview.innerHTML = patternHtml;
        patternPreview.style.display = 'block';
        
        // Animate pattern appearance
        const svg = patternPreview.querySelector('svg');
        if (svg) {
            svg.style.opacity = '0';
            svg.style.transform = 'scale(0.8)';
            
            setTimeout(() => {
                svg.style.transition = 'all 0.6s ease';
                svg.style.opacity = '1';
                svg.style.transform = 'scale(1)';
            }, 100);
        }
    }
}

// Generate similar pattern based on classification
async function generateSimilarPattern() {
    if (!classificationResult) {
        showAlert('No classification result available', 'warning');
        return;
    }
    
    const parameters = {
        pattern_type: classificationResult.classification,
        grid_size: 12,
        symmetry: 'rotational',
        complexity: 6,
        color_scheme: 'traditional'
    };
    
    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(parameters)
        });
        
        const result = await response.json();
        if (result.success) {
            displayGeneratedPattern(result);
        }
        
    } catch (error) {
        console.error('Similar pattern generation error:', error);
        showAlert('Failed to generate similar pattern', 'error');
    }
}

// Update slider value display
function updateSliderValue(e) {
    const slider = e.target;
    const valueDisplay = document.querySelector(`#${slider.id}-value`);
    if (valueDisplay) {
        valueDisplay.textContent = slider.value;
    }
}

// Preview pattern changes in real-time
function previewPatternChanges() {
    // Debounce the preview updates
    clearTimeout(window.previewTimeout);
    window.previewTimeout = setTimeout(() => {
        // This would trigger a lightweight preview generation
        console.log('Previewing pattern changes...');
    }, 500);
}

// Handle file downloads
async function handleDownload(format) {
    if (!generatedPattern) {
        showAlert('No pattern to download', 'warning');
        return;
    }
    
    try {
        let downloadUrl;
        let filename;
        
        if (format === 'svg') {
            const blob = new Blob([generatedPattern.svg_content], { type: 'image/svg+xml' });
            downloadUrl = URL.createObjectURL(blob);
            filename = `kolam_pattern_${Date.now()}.svg`;
        } else if (format === 'png') {
            // Convert SVG to PNG using canvas
            downloadUrl = await convertSvgToPng(generatedPattern.svg_content);
            filename = `kolam_pattern_${Date.now()}.png`;
        }
        
        // Trigger download
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Clean up object URL
        URL.revokeObjectURL(downloadUrl);
        
        showAlert(`Pattern downloaded as ${format.toUpperCase()}`, 'success');
        
    } catch (error) {
        console.error('Download error:', error);
        showAlert('Failed to download pattern', 'error');
    }
}

// Convert SVG to PNG
async function convertSvgToPng(svgContent) {
    return new Promise((resolve, reject) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        img.onload = function() {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            
            canvas.toBlob((blob) => {
                if (blob) {
                    resolve(URL.createObjectURL(blob));
                } else {
                    reject(new Error('Failed to convert SVG to PNG'));
                }
            }, 'image/png');
        };
        
        img.onerror = () => reject(new Error('Failed to load SVG'));
        
        const svgBlob = new Blob([svgContent], { type: 'image/svg+xml' });
        img.src = URL.createObjectURL(svgBlob);
    });
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showLoading(message = 'Loading...') {
    if (loadingOverlay) {
        loadingOverlay.querySelector('.loading-message').textContent = message;
        loadingOverlay.style.display = 'flex';
    }
}

function hideLoading() {
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
}

function showAlert(message, type = 'info') {
    // Create alert element
    const alertHtml = `
        <div class="alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    // Find or create alert container
    let alertContainer = document.getElementById('alert-container');
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.id = 'alert-container';
        alertContainer.className = 'position-fixed top-0 end-0 p-3';
        alertContainer.style.zIndex = '9999';
        document.body.appendChild(alertContainer);
    }
    
    alertContainer.insertAdjacentHTML('afterbegin', alertHtml);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        const alert = alertContainer.querySelector('.alert');
        if (alert) {
            alert.classList.remove('show');
            setTimeout(() => alert.remove(), 150);
        }
    }, 5000);
}

function shareResult() {
    if (!classificationResult) return;
    
    const shareText = `Check out my Kolam classification result: ${classificationResult.classification.replace('_', ' ')} with ${(classificationResult.confidence * 100).toFixed(1)}% confidence!`;
    
    if (navigator.share) {
        navigator.share({
            title: 'Kolam AI Classification Result',
            text: shareText,
            url: window.location.href
        });
    } else {
        // Fallback: copy to clipboard
        navigator.clipboard.writeText(shareText).then(() => {
            showAlert('Result copied to clipboard!', 'success');
        });
    }
}

// Initialize tooltips (if Bootstrap is available)
function initializeTooltips() {
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

// Initialize animations
function initializeAnimations() {
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    const animateElements = document.querySelectorAll('.stats-card, .pattern-item, .result-card');
    animateElements.forEach(el => observer.observe(el));
}

// Form validation
function setupFormValidation() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!form.checkValidity()) {
                e.preventDefault();
                e.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
}

// Modal functionality
function showModal(e) {
    e.preventDefault();
    const modalId = e.target.dataset.target;
    const modal = document.querySelector(modalId);
    if (modal && typeof bootstrap !== 'undefined') {
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
    }
}

function showGalleryModal(e) {
    const imageSrc = e.target.src;
    const imageAlt = e.target.alt;
    
    // Create or update gallery modal
    let modal = document.getElementById('gallery-modal');
    if (!modal) {
        const modalHtml = `
            <div class="modal fade" id="gallery-modal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-body p-0">
                            <img id="gallery-modal-img" class="img-fluid w-100" src="" alt="">
                        </div>
                    </div>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        modal = document.getElementById('gallery-modal');
    }
    
    const modalImg = document.getElementById('gallery-modal-img');
    modalImg.src = imageSrc;
    modalImg.alt = imageAlt;
    
    if (typeof bootstrap !== 'undefined') {
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
    }
}

// Export functions for global access
window.uploadImage = uploadImage;
window.classifyImage = classifyImage;
window.generateSimilarPattern = generateSimilarPattern;
window.shareResult = shareResult;