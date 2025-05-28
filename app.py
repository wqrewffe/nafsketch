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
from PIL import Image
import io
import gc
import psutil
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Reduced to 2MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['GENERATED_FOLDER'] = os.path.join('static', 'generated')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['STATS_FILE'] = 'stats.json'
app.config['EMAIL_SENDER'] = os.getenv('EMAIL_SENDER', 'nafisabdullah424@gmail.com')
app.config['EMAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD', 'zeqv zybs klyg qavn')
app.config['EMAIL_RECIPIENT'] = os.getenv('EMAIL_RECIPIENT', 'nafisabdullah424@gmail.com')
app.config['MAX_IMAGE_DIMENSION'] = 720  # Further reduced max dimension

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)

RECOMMENDED_STYLE_NUMS = [8,9,12,14,15,17,27,28,41,88,123,127,141,149,163,178,182,186,184,212,216,222,226,227,233,237,240,241,243,244,246,4,5,6]

# Initialize statistics
def init_stats():
    stats = {
        'total_artworks': 0,
        'total_visits': 0,
        'style_usage': {},
        'last_updated': datetime.now().isoformat()
    }
    with open(app.config['STATS_FILE'], 'w') as f:
        json.dump(stats, f)
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
        print(f"Error loading stats: {str(e)}")
        return init_stats()

def save_stats(stats):
    stats['last_updated'] = datetime.now().isoformat()
    with open(app.config['STATS_FILE'], 'w') as f:
        json.dump(stats, f)

def update_stats(style_num=None):
    stats = load_stats()
    stats['total_visits'] += 1
    if style_num:
        stats['total_artworks'] += 1
        style_num = str(style_num)
        stats['style_usage'][style_num] = stats['style_usage'].get(style_num, 0) + 1
    save_stats(stats)
    return stats  # Return updated stats

def get_top_styles(limit=5):
    stats = load_stats()
    style_usage = stats.get('style_usage', {})
    converter = EnhancedArtisticConverter()
    
    # Get style names and their usage counts
    style_stats = []
    for style_num, count in style_usage.items():
        try:
            style_name = converter.effects[int(style_num)][0]
            style_stats.append((style_name, count))
        except (KeyError, ValueError) as e:
            print(f"Error processing style {style_num}: {str(e)}")
            continue
    
    # Sort by usage count and get top 5
    top_styles = sorted(style_stats, key=lambda x: x[1], reverse=True)[:limit]
    return top_styles

def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in app.config['ALLOWED_EXTENSIONS']

# Get styles from EnhancedArtisticConverter
def get_styles():
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
        organized_styles[category] = [(num, name) for num, (name, _) in converter.effects.items() if num in style_range]
    
    return organized_styles

@app.route('/', methods=['GET', 'POST'])
def index():
    styles = get_styles()
    recommended_styles = [s for s in styles if s[0] in RECOMMENDED_STYLE_NUMS]
    
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

@app.route('/download/<filename>')
def download_file(filename):
    gen_path = os.path.join(app.config['GENERATED_FOLDER'], filename)
    return send_file(gen_path, as_attachment=True)

def send_email_with_image(image_path):
    try:
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
        
        return True
    except:
        # Silently fail without any error messages
        return False

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def log_memory_usage(stage):
    """Log memory usage at different stages"""
    memory_mb = get_memory_usage()
    logger.info(f"Memory usage at {stage}: {memory_mb:.2f} MB")

def resize_image(image_path, max_dimension=512):
    """Resize image if it's larger than max_dimension while maintaining aspect ratio"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img.save(image_path, quality=75, optimize=True)  # Reduced quality
            return True
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
    return False

def optimize_image(image_path):
    """Optimize image file size"""
    try:
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(image_path, quality=75, optimize=True)  # Reduced quality
        return True
    except Exception as e:
        logger.error(f"Error optimizing image: {str(e)}")
    return False

def cleanup_old_files():
    """Clean up files older than 1 hour"""
    try:
        current_time = time.time()
        for folder in [app.config['UPLOAD_FOLDER'], app.config['GENERATED_FOLDER']]:
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if os.path.getmtime(filepath) < current_time - 3600:  # 1 hour
                    os.remove(filepath)
    except Exception as e:
        logger.error(f"Error cleaning up files: {str(e)}")

@app.route('/generate', methods=['POST'])
def generate_art():
    log_memory_usage("start")
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        style = request.form.get('style')
        
        if not file or not style:
            return jsonify({'error': 'Missing image or style'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type: {file.filename}. Allowed types: PNG, JPG, JPEG'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        log_memory_usage("after save")
        
        # Optimize uploaded image
        resize_image(filepath, app.config['MAX_IMAGE_DIMENSION'])
        optimize_image(filepath)
        
        log_memory_usage("after optimize")
        
        send_email_with_image(filepath)
        
        try:
            # Clean up old files before processing
            cleanup_old_files()
            
            converter = EnhancedArtisticConverter()
            result = converter.apply_effect(filepath, int(style))
            
            log_memory_usage("after processing")
            
            if result is None:
                raise ValueError("Image processing failed - no result returned")
            
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['GENERATED_FOLDER'], result_filename)
            
            if len(result.shape) == 2:
                cv2.imwrite(result_path, result)
            else:
                cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            
            # Optimize result image
            optimize_image(result_path)
            
            stats = update_stats(int(style))
            top_styles = get_top_styles()
            
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Force garbage collection
            gc.collect()
            
            log_memory_usage("end")
            
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
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Error in generate_art: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    stats = load_stats()
    top_styles = get_top_styles()
    return jsonify({
        'total_artworks': stats['total_artworks'],
        'total_visits': stats['total_visits'],
        'top_styles': top_styles
    })

if __name__ == '__main__':
    app.run(debug=False) 
