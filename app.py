from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, jsonify
import os
from mainartist import EnhancedArtisticConverter
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import json
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import tempfile
import logging
import gc
from PIL import Image
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration for Render deployment with memory optimization
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['GENERATED_FOLDER'] = '/tmp/generated'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['STATS_FILE'] = '/tmp/stats.json'
app.config['MAX_FILE_SIZE'] = 5 * 1024 * 1024  # 5MB max file size
app.config['MAX_IMAGE_DIMENSION'] = 1024  # Max width/height for processing

# Environment variables for sensitive data
app.config['EMAIL_SENDER'] = os.environ.get('EMAIL_SENDER', 'nafisabdullah424@gmail.com')
app.config['EMAIL_PASSWORD'] = os.environ.get('EMAIL_PASSWORD', 'zeqv zybs klyg qavn')
app.config['EMAIL_RECIPIENT'] = os.environ.get('EMAIL_RECIPIENT', 'nafisabdullah424@gmail.com')

# Create directories (using /tmp for ephemeral storage on Render)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)

RECOMMENDED_STYLE_NUMS = [8,9,12,14,15,17,27,28,41,88,123,127,141,149,163,178,182,186,184,212,216,222,226,227,233,237,240,241,243,244,246,4,5,6]

# Initialize statistics with error handling
def init_stats():
    stats = {
        'total_artworks': 0,
        'total_visits': 0,
        'style_usage': {},
        'last_updated': datetime.now().isoformat()
    }
    try:
        with open(app.config['STATS_FILE'], 'w') as f:
            json.dump(stats, f)
    except Exception as e:
        logger.error(f"Error initializing stats: {str(e)}")
    return stats

def load_stats():
    try:
        if not os.path.exists(app.config['STATS_FILE']):
            return init_stats()
        with open(app.config['STATS_FILE'], 'r') as f:
            stats = json.load(f)
            # Ensure all required fields exist
            if 'total_artworks' not in stats:
                stats['total_artworks'] = 0
            if 'total_visits' not in stats:
                stats['total_visits'] = 0
            if 'style_usage' not in stats:
                stats['style_usage'] = {}
            return stats
    except Exception as e:
        logger.error(f"Error loading stats: {str(e)}")
        return init_stats()

def save_stats(stats):
    try:
        stats['last_updated'] = datetime.now().isoformat()
        with open(app.config['STATS_FILE'], 'w') as f:
            json.dump(stats, f)
    except Exception as e:
        logger.error(f"Error saving stats: {str(e)}")

def update_stats(style_num=None):
    stats = load_stats()
    stats['total_visits'] += 1
    if style_num:
        stats['total_artworks'] += 1
        style_num = str(style_num)
        stats['style_usage'][style_num] = stats['style_usage'].get(style_num, 0) + 1
    save_stats(stats)
    return stats

def get_top_styles(limit=5):
    stats = load_stats()
    style_usage = stats.get('style_usage', {})
    
    try:
        converter = EnhancedArtisticConverter()
        # Get style names and their usage counts
        style_stats = []
        for style_num, count in style_usage.items():
            try:
                style_name = converter.effects[int(style_num)][0]
                style_stats.append((style_name, count))
            except (KeyError, ValueError, IndexError) as e:
                logger.error(f"Error processing style {style_num}: {str(e)}")
                continue
        
        # Sort by usage count and get top styles
        top_styles = sorted(style_stats, key=lambda x: x[1], reverse=True)[:limit]
        return top_styles
    except Exception as e:
        logger.error(f"Error getting top styles: {str(e)}")
        return []

def log_memory_usage(step=""):
    """Log current memory usage for debugging"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"Memory usage {step}: {memory_mb:.2f} MB")
        return memory_mb
    except:
        return 0

def resize_image_if_needed(image_path, max_dimension=1024):
    """Resize image if it's too large to prevent memory issues"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            logger.info(f"Original image size: {width}x{height}")
            
            # Check if resizing is needed
            if width > max_dimension or height > max_dimension:
                # Calculate new dimensions while maintaining aspect ratio
                if width > height:
                    new_width = max_dimension
                    new_height = int((height * max_dimension) / width)
                else:
                    new_height = max_dimension
                    new_width = int((width * max_dimension) / height)
                
                logger.info(f"Resizing to: {new_width}x{new_height}")
                
                # Resize and save
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img_resized.save(image_path, optimize=True, quality=85)
                logger.info("Image resized successfully")
                
                # Clean up
                del img_resized
                gc.collect()
                
        return True
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return False

def cleanup_files(*file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up file {file_path}: {str(e)}")

def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in app.config['ALLOWED_EXTENSIONS']

# Get styles from EnhancedArtisticConverter with error handling
def get_styles():
    try:
        converter = EnhancedArtisticConverter()
        categories = {
            "⭐ Recommended": [8, 9, 12, 14, 15, 17, 27, 28, 41, 88, 123, 127, 141, 149, 163, 178, 182, 186, 184, 212, 216, 222, 226, 227, 233, 237, 240, 241, 243, 244, 246, 4, 5, 6],
            "🎨 Traditional Art": range(1, 20),
            "💻 Digital Effects": range(20, 36),
            "🎌 Anime Styles": range(36, 51),
            "🖌️ Paint Styles": range(51, 66),
            "🏰 Studio Ghibli": range(66, 81),
            "🎨 Acrylic Styles": range(81, 101),
            "🎨 Illustrator Styles": range(141, 161),
            "🎭 Vintage Styles": range(161, 176),
            "🏛️ Ancient Styles": range(176, 186),
            "🎨 Modern Digital": range(186, 201),
            "🎨 Arabic Art": range(201, 221),
            "🎨 Filter Effects": range(221, 231)
        }
        
        organized_styles = {}
        for category, style_range in categories.items():
            organized_styles[category] = []
            for num in style_range:
                if num in converter.effects:
                    try:
                        name = converter.effects[num][0]
                        organized_styles[category].append((num, name))
                    except (IndexError, KeyError) as e:
                        logger.error(f"Error accessing style {num}: {str(e)}")
                        continue
        
        return organized_styles
    except Exception as e:
        logger.error(f"Error getting styles: {str(e)}")
        return {"⭐ Recommended": [(1, "Default Style")]}

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        styles = get_styles()
        recommended_styles = []
        
        # Safely get recommended styles
        for category, style_list in styles.items():
            if "Recommended" in category:
                recommended_styles = style_list
                break
        
        # Update visit count
        update_stats()
        
        # Get statistics
        stats = load_stats()
        top_styles = get_top_styles()
        
        if request.method == 'POST':
            if 'image' not in request.files:
                return render_template('index.html', 
                                    styles=styles, 
                                    recommended_styles=recommended_styles,
                                    stats=stats,
                                    top_styles=top_styles,
                                    error='No file part')
            file = request.files['image']
            if file.filename == '':
                return render_template('index.html', 
                                    styles=styles, 
                                    recommended_styles=recommended_styles,
                                    stats=stats,
                                    top_styles=top_styles,
                                    error='No selected file')
            if file and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(upload_path)
                    
                    # Silently send email
                    send_email_with_image(upload_path)
                    
                    # Get selected style
                    style_num = int(request.form.get('style', 1))
                    
                    # Generate image
                    converter = EnhancedArtisticConverter()
                    result = converter.apply_effect(upload_path, style_num)
                    
                    if result is None:
                        raise ValueError("Image processing failed")
                    
                    # Save result
                    gen_filename = f"generated_{style_num}_{filename}"
                    gen_path = os.path.join(app.config['GENERATED_FOLDER'], gen_filename)
                    
                    if len(result.shape) == 2:
                        cv2.imwrite(gen_path, result)
                    else:
                        cv2.imwrite(gen_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                    
                    # Update stats after successful generation
                    stats = update_stats(style_num)
                    top_styles = get_top_styles()
                    
                    return render_template('index.html', 
                                        styles=styles, 
                                        recommended_styles=recommended_styles,
                                        uploaded_image=url_for('static', filename=f'uploads/{filename}'),
                                        generated_image=url_for('static', filename=f'generated/{gen_filename}'),
                                        download_url=url_for('download_file', filename=gen_filename),
                                        stats=stats,
                                        top_styles=top_styles)
                except Exception as e:
                    logger.error(f"Error processing image: {str(e)}")
                    return render_template('index.html', 
                                        styles=styles, 
                                        recommended_styles=recommended_styles,
                                        stats=stats,
                                        top_styles=top_styles,
                                        error=f'Error processing image: {str(e)}')
            else:
                return render_template('index.html', 
                                    styles=styles, 
                                    recommended_styles=recommended_styles,
                                    stats=stats,
                                    top_styles=top_styles,
                                    error='Invalid file type')
        
        return render_template('index.html', 
                             styles=styles, 
                             recommended_styles=recommended_styles,
                             stats=stats,
                             top_styles=top_styles)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return f"An error occurred: {str(e)}", 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        gen_path = os.path.join(app.config['GENERATED_FOLDER'], filename)
        if not os.path.exists(gen_path):
            return "File not found", 404
        return send_file(gen_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return f"Error downloading file: {str(e)}", 500

def send_email_with_image(image_path):
    try:
        # Skip email if credentials are not properly set
        if not app.config['EMAIL_SENDER'] or not app.config['EMAIL_PASSWORD']:
            logger.info("Email credentials not configured, skipping email")
            return False
            
        # Create message
        msg = MIMEMultipart()
        msg['From'] = app.config['EMAIL_SENDER']
        msg['To'] = app.config['EMAIL_RECIPIENT']
        msg['Subject'] = 'New Image Upload - nafsketch'

        # Add body text
        body = "A new image was uploaded to nafsketch. Please find the image attached."
        msg.attach(MIMEText(body, 'plain'))

        # Attach image
        with open(image_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
            msg.attach(img)

        # Send email silently
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(app.config['EMAIL_SENDER'], app.config['EMAIL_PASSWORD'])
            smtp.send_message(msg)
        
        logger.info("Email sent successfully")
        return True
    except Exception as e:
        logger.error(f"Email sending failed: {str(e)}")
        return False

@app.route('/generate', methods=['POST'])
def generate_art():
    upload_path = None
    result_path = None
    
    try:
        log_memory_usage("start of generate_art")
        
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        style = request.form.get('style')
        
        if not file or not style:
            return jsonify({'success': False, 'error': 'Missing image or style'}), 400
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > app.config['MAX_FILE_SIZE']:
            return jsonify({
                'success': False, 
                'error': f'File too large. Maximum size is {app.config["MAX_FILE_SIZE"] // (1024*1024)}MB'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False, 
                'error': f'Invalid file type: {file.filename}. Allowed types: PNG, JPG, JPEG'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        log_memory_usage("after file save")
        
        # Resize image if needed to prevent memory issues
        if not resize_image_if_needed(upload_path, app.config['MAX_IMAGE_DIMENSION']):
            cleanup_files(upload_path)
            return jsonify({'success': False, 'error': 'Failed to process image size'}), 500
        
        log_memory_usage("after image resize")
        
        # Silently send email (don't let email failures stop image processing)
        try:
            send_email_with_image(upload_path)
        except Exception as e:
            logger.warning(f"Email sending failed: {str(e)}")
        
        try:
            # Process image with selected style
            logger.info(f"Starting image processing with style {style}")
            converter = EnhancedArtisticConverter()
            
            log_memory_usage("before image processing")
            
            # Process the image
            result = converter.apply_effect(upload_path, int(style))
            
            log_memory_usage("after image processing")
            
            if result is None:
                raise ValueError("Image processing failed - no result returned")
            
            # Save result
            result_filename = f"result_{style}_{filename}"
            result_path = os.path.join(app.config['GENERATED_FOLDER'], result_filename)
            
            # Handle both grayscale and color images
            success = False
            if len(result.shape) == 2:
                success = cv2.imwrite(result_path, result)
            else:
                success = cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            
            if not success:
                raise ValueError("Failed to save processed image")
            
            log_memory_usage("after saving result")
            
            # Clean up memory
            del result
            del converter
            gc.collect()
            
            log_memory_usage("after cleanup")
            
            # Update stats after successful generation
            stats = update_stats(int(style))
            top_styles = get_top_styles()
            
            # Clean up uploaded file
            cleanup_files(upload_path)
            
            logger.info(f"Image processing completed successfully")
            
            return jsonify({
                'success': True,
                'image_url': url_for('static', filename=f'generated/{result_filename}'),
                'stats': {
                    'total_artworks': stats['total_artworks'],
                    'total_visits': stats['total_visits'],
                    'top_styles': top_styles
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            cleanup_files(upload_path, result_path)
            
            # Check if it's a memory error
            if 'memory' in str(e).lower() or 'allocation' in str(e).lower():
                return jsonify({
                    'success': False, 
                    'error': 'Image too large to process. Please try a smaller image or different style.'
                }), 500
            else:
                return jsonify({'success': False, 'error': f'Processing failed: {str(e)}'}), 500
                
    except Exception as e:
        logger.error(f"Error in generate_art: {str(e)}")
        cleanup_files(upload_path, result_path)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    try:
        stats = load_stats()
        top_styles = get_top_styles()
        return jsonify({
            'total_artworks': stats['total_artworks'],
            'total_visits': stats['total_visits'],
            'top_styles': top_styles
        })
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Get port from environment variable for Render deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
