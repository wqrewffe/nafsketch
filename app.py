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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration for Render deployment
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['GENERATED_FOLDER'] = '/tmp/generated'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['STATS_FILE'] = '/tmp/stats.json'

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
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        style = request.form.get('style')
        
        if not file or not style:
            return jsonify({'success': False, 'error': 'Missing image or style'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': f'Invalid file type: {file.filename}. Allowed types: PNG, JPG, JPEG'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Silently send email
        send_email_with_image(filepath)
        
        try:
            # Process image with selected style
            converter = EnhancedArtisticConverter()
            result = converter.apply_effect(filepath, int(style))
            
            if result is None:
                raise ValueError("Image processing failed - no result returned")
            
            # Save result
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['GENERATED_FOLDER'], result_filename)
            
            # Handle both grayscale and color images
            if len(result.shape) == 2:
                cv2.imwrite(result_path, result)
            else:
                cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            
            # Update stats after successful generation
            stats = update_stats(int(style))
            top_styles = get_top_styles()
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except Exception as e:
                logger.warning(f"Could not remove uploaded file: {str(e)}")
            
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
            return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Error in generate_art: {str(e)}")
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
