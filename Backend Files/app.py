from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import sys
import cv2
import numpy as np
import json
from datetime import datetime

# Add current directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from pose_analyzer import PoseAnalyzer
from video_processor import VideoProcessor

# Get the project root (parent of Backend Files folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Setup paths
TEMPLATE_FOLDER = os.path.join(PROJECT_ROOT, 'Frontend', 'templates')
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads')
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'outputs')

# Verify template folder exists
if not os.path.exists(TEMPLATE_FOLDER):
    print(f"❌ Template folder not found: {TEMPLATE_FOLDER}")
    print(f"Creating template folder...")
    os.makedirs(TEMPLATE_FOLDER, exist_ok=True)
else:
    print(f"✓ Template folder found: {TEMPLATE_FOLDER}")

# Check if index.html exists
index_path = os.path.join(TEMPLATE_FOLDER, 'index.html')
if not os.path.exists(index_path):
    print(f"❌ index.html not found at: {index_path}")
    print(f"Please ensure index.html is in the Frontend/templates/ folder")
else:
    print(f"✓ index.html found")

# Create Flask app with custom template folder
app = Flask(__name__, 
            template_folder=TEMPLATE_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

print("\n" + "="*60)
print("DeadLift Pro - Configuration")
print("="*60)
print(f"Base directory: {BASE_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Template folder: {TEMPLATE_FOLDER}")
print(f"Upload folder: {UPLOAD_FOLDER}")
print(f"Output folder: {OUTPUT_FOLDER}")
print("="*60 + "\n")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image or video"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save uploaded file
        file.save(filepath)
        print(f"\n✓ File uploaded: {unique_filename}")
        
        # Determine if it's an image or video
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        try:
            if file_ext in ['jpg', 'jpeg', 'png']:
                # Process image
                print(f"📸 Processing image...")
                analyzer = PoseAnalyzer()
                result = analyzer.analyze_image(filepath)
                
                # Save annotated image
                output_filename = f"analyzed_{unique_filename}"
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                cv2.imwrite(output_path, result['annotated_frame'])
                
                print(f"✓ Image analysis complete!")
                print(f"  Score: {result['score']*100:.1f}%")
                print(f"  Output: {output_filename}\n")
                
                return jsonify({
                    'success': True,
                    'type': 'image',
                    'output_file': output_filename,
                    'score': result['score'],
                    'feedback': result['feedback'],
                    'keypoints': result['keypoints'],
                    'metrics': result['metrics']
                })
            else:
                # Process video
                print(f"🎥 Processing video...")
                processor = VideoProcessor()
                output_filename = f"analyzed_{unique_filename.rsplit('.', 1)[0]}.mp4"
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                
                result = processor.process_video(filepath, output_path)
                
                print(f"✓ Video analysis complete!")
                print(f"  Average Score: {result['avg_score']*100:.1f}%")
                print(f"  Frames: {result['frame_count']}")
                print(f"  Output: {output_filename}\n")
                
                return jsonify({
                    'success': True,
                    'type': 'video',
                    'output_file': output_filename,
                    'frame_scores': result['frame_scores'],
                    'avg_score': result['avg_score'],
                    'feedback': result['feedback'],
                    'frame_count': result['frame_count']
                })
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error during analysis: {error_msg}\n")
            return jsonify({'error': error_msg}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/outputs/<filename>')
def output_file(filename):
    """Serve analyzed output files"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/history')
def history():
    """Get list of analyzed files"""
    files = []
    
    if os.path.exists(app.config['OUTPUT_FOLDER']):
        for filename in os.listdir(app.config['OUTPUT_FOLDER']):
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            if os.path.isfile(filepath):
                files.append({
                    'name': filename,
                    'timestamp': datetime.fromtimestamp(os.path.getctime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                })
    
    files.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify({'files': files})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'template_folder': TEMPLATE_FOLDER,
        'template_exists': os.path.exists(index_path),
        'upload_folder': UPLOAD_FOLDER,
        'output_folder': OUTPUT_FOLDER
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏋️  DeadLift Pro - Starting Server")
    print("="*60)
    
    # Verify model files exist
    model_proto = os.path.join(PROJECT_ROOT, 'Model Files', 'pose', 'mpi', 'pose_deploy_linevec_faster_4_stages.prototxt')
    model_weights = os.path.join(PROJECT_ROOT, 'Model Files', 'pose', 'mpi', 'pose_iter_160000.caffemodel')
    
    if not os.path.exists(model_proto):
        print(f"⚠️  Warning: Model prototxt not found at:")
        print(f"   {model_proto}")
        print(f"   Please download OpenPose model files!")
    else:
        print(f"✓ Model prototxt found")
    
    if not os.path.exists(model_weights):
        print(f"⚠️  Warning: Model weights not found at:")
        print(f"   {model_weights}")
        print(f"   Please download OpenPose model files!")
    else:
        print(f"✓ Model weights found")
    
    print("\n" + "="*60)
    print("🚀 Server starting at: http://localhost:5000")
    print("="*60)
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)