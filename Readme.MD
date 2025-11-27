# 🏋️ DeadLift Pro - AI Form Analysis

**Developed by V NAVNEETH**

A professional web application for analyzing deadlift form using computer vision and pose estimation.

---

## ✨ Features

- **Real-time Pose Detection**: Uses OpenPose MPI model for accurate keypoint detection
- **Image & Video Analysis**: Supports both static images and video files
- **Detailed Metrics**: Calculates back angle, hip hinge, knee position, and bar path
- **Visual Feedback**: Annotated output with skeleton overlay and form scores
- **Modern UI**: Beautiful, responsive interface with drag-and-drop upload
- **Analysis History**: Track your progress over time
- **Portfolio Ready**: Professional design suitable for showcasing

## 📋 Prerequisites

- Python 3.8 or higher
- OpenPose model files (MPI)
- Webcam or video/image files for analysis

## 🚀 Installation

### 1. Clone or Download the Project

```bash
# Create project directory
mkdir deadlift-pro
cd deadlift-pro
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```


### 4. Project Structure

Organize your files as follows:

```
deadlift-pro/
├── app.py                  # Main Flask application
├── pose_analyzer.py        # Pose analysis core
├── video_processor.py      # Video processing module
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Web interface
├── pose/
│   └── mpi/
│       ├── pose_deploy_linevec_faster_4_stages.prototxt
│       └── pose_iter_160000.caffemodel
├── uploads/               # Created automatically
├── outputs/               # Created automatically
└── good/                  # Optional: training images (good form)
    └── bad/               # Optional: training images (bad form)
```

### 5. (Optional) Add Training Data

If you have labeled training data:

1. Place good form images in `good/` folder (1.jpg, 2.jpg, etc.)
2. Place bad form images in `bad/` folder (1.jpg, 2.jpg, etc.)
3. The system will use these to improve predictions

## 🎮 Usage

### Starting the Server

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Using the Application

1. **Open your browser** and navigate to `http://localhost:5000`
2. **Upload a file**: Drag and drop or click to browse
   - Supported formats: MP4, AVI, MOV, JPG, PNG
3. **Click "Analyze Form"** to start the analysis
4. **Review results**: 
   - Form score (0-100%)
   - Detailed feedback
   - Annotated visualization
   - Key metrics (angles, positions)

### Features

- **Drag & Drop**: Simply drag your file onto the upload area
- **Real-time Preview**: See your uploaded file before analysis
- **Detailed Metrics**: 
  - Back angle (spine neutrality)
  - Hip hinge angle
  - Knee position
  - Bar path alignment
- **Visual Feedback**: Color-coded skeleton overlay
  - Green: Excellent form
  - Orange: Good form with minor issues
  - Red: Poor form, needs improvement
- **History**: Click the 📊 button to view past analyses

## 🔧 Configuration

### Adjust Sensitivity

Edit `pose_analyzer.py` to modify thresholds:

```python
# Line 53: Detection threshold
self.threshold = 0.1  # Lower = more lenient

# Line 119: Back angle range (good form)
if 160 <= back_angle <= 200:  # Adjust these values

# Line 125: Hip hinge angle range
if 30 <= hip_angle <= 90:

# Line 131: Knee angle range
if 160 <= knee_angle <= 180:
```

### Video Processing Speed

Edit `app.py` for video frame rate:

```python
# Line 11: Max file size
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
```

## 📊 Understanding the Metrics

### Form Score
- **80-100%**: Excellent form, maintain this!
- **60-79%**: Good form with minor improvements needed
- **0-59%**: Form needs significant attention

### Back Angle
- Optimal: 160-200°
- Indicates spine neutrality
- Lower values suggest excessive rounding

### Hip Hinge
- Optimal: 30-90°
- Shows proper hip movement pattern
- Critical for deadlift mechanics

### Knee Angle
- Optimal: 160-180°
- Slight bend is good
- Locked knees are problematic

### Bar Path
- Should be close to body
- Over midfoot throughout lift
- Measured as distance from wrists to ankles

## 🎯 Tips for Best Results

1. **Camera Position**: Side view is best for analysis
2. **Lighting**: Ensure good, even lighting
3. **Full Body**: Make sure entire body is visible
4. **Contrast**: Wear contrasting clothes against background
5. **Video Quality**: Higher resolution = better detection
6. **Stable Camera**: Minimize camera movement

## 🔍 Troubleshooting

### "No keypoints detected"
- Ensure full body is visible in frame
- Check lighting conditions
- Try adjusting threshold in `pose_analyzer.py`

### Slow video processing
- Reduce video resolution
- Process shorter clips
- Close other applications

### Model files not found
- Verify `pose/mpi/` folder structure
- Check file names match exactly
- Re-download model files if needed

## 🚀 Advanced Features

### Custom Training

To train with your own data:

1. Collect 10+ images each of good and bad form
2. Place in `good/` and `bad/` folders
3. Run the training section from your original notebook
4. The classifier will automatically improve

### API Integration

The app exposes REST endpoints:

```python
# Analyze endpoint
POST /analyze
Content-Type: multipart/form-data
Body: file=<your_file>

# History endpoint
GET /history
Returns: JSON list of analyzed files
```

### Batch Processing

Modify `video_processor.py` to process multiple files:

```python
# Add this method to VideoProcessor class
def batch_process(self, input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(('.mp4', '.avi')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"analyzed_{filename}")
            self.process_video(input_path, output_path)
```


## 📧 Support

For issues or questions, please refer to the troubleshooting section or check the code comments.

---

## 👨‍💻 Author

**V NAVNEETH**

This project demonstrates expertise in:
- Computer Vision & Deep Learning
- Full-Stack Web Development
- OpenPose & Pose Estimation
- Real-time Video Processing
- User Interface Design
- Python & Flask Development


**Made with ❤️ by V NAVNEETH for better lifting form and safer training**

