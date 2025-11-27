# 🏋️ DeadLift Pro - AI Form Analysis System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**AI-Powered Deadlift Form Analysis using Computer Vision**
</div>

## 📋 Overview

DeadLift Pro is a professional-grade web application that analyzes deadlift form in real-time using OpenPose and machine learning. It provides instant feedback on form quality, detailed biomechanical metrics, and tracks progress over time.

### 🎯 Key Capabilities
- **Real-time Pose Detection** - 15 body keypoints tracked
- **Form Scoring** - AI-powered 0-100% accuracy rating
- **Video & Image Analysis** - Process any lifting footage
- **Biomechanical Metrics** - Back angle, hip hinge, knee position, bar path
- **Progress Tracking** - Monitor improvements over time
- **Export Reports** - Save and share analysis results

---

## ✨ Features

### 🤖 AI-Powered Analysis
- OpenPose MPI model integration
- Real-time skeleton detection
- Machine learning form classification
- Severity-based scoring algorithm

### 📊 Advanced Analytics
- **Frame-by-Frame Analysis** - Video breakdown with score charts
- **Comparison Mode** - Side-by-side progress comparison
- **Progress Tracker** - Long-term improvement monitoring
- **Detailed Metrics** - Angles, distances, alignment data

### 🎨 Modern Interface
- Clean, responsive design
- Dark theme with gradient accents
- Interactive data visualization
- Tab-based navigation (5 sections)
- Mobile-friendly layout

### 💾 Smart Features
- LocalStorage data persistence
- Analysis history tracking
- Export functionality (PDF, Share)
- Pro tips and educational content

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 2GB free disk space (for model files)

### Quick Start

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/deadlift-pro.git
cd deadlift-pro
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download OpenPose Models**

Download these files and place in `Model Files/pose/mpi/`:
- [pose_deploy_linevec_faster_4_stages.prototxt](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt)
- [pose_iter_160000.caffemodel](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel) (1.5 GB)

4. **Run the Application**
```bash
cd "Backend Files"
python app.py
```

5. **Open in Browser**
```
http://localhost:5000
```

---

## 📁 Project Structure

```
deadlift-pro/
├── Backend Files/
│   ├── app.py                    # Flask server
│   ├── pose_analyzer.py          # Pose detection & analysis
│   ├── video_processor.py        # Video processing
│   ├── config.py                 # Configuration
│   └── train_model.py            # Model training script
├── Frontend/
│   └── templates/
│       └── index.html            # Web interface
├── Model Files/
│   └── pose/
│       └── mpi/
│           ├── *.prototxt        # Model architecture
│           └── *.caffemodel      # Pre-trained weights
├── Training Data/
│   ├── good/                     # Good form examples
│   └── bad/                      # Bad form examples
├── uploads/                      # User uploads (auto-created)
├── outputs/                      # Analysis results (auto-created)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── LICENSE                       # MIT License
└── .gitignore                    # Git ignore rules
```

---

## 🎮 Usage

### 1. Upload & Analyze
- Drag and drop or browse for video/image
- Supported formats: MP4, AVI, MOV, JPG, PNG
- Click "Analyze Form" button

### 2. View Results
- Form score (0-100%)
- Color-coded feedback (Green/Orange/Red)
- Detailed metrics with angles
- Annotated output with skeleton overlay

### 3. Track Progress
- Navigate to "Progress Tracker" tab
- View improvement in each metric
- Check session statistics
- Monitor consistency

### 4. Compare Results
- Use "Compare" tab
- See before/after improvements
- Track best scores
- Analyze trends

### 5. Learn Best Practices
- Visit "Pro Tips" section
- Recording guidelines
- Form fundamentals
- Common mistakes

---

## 📊 Technical Details

### Pose Detection
- **Model**: OpenPose MPI (15 keypoints)
- **Resolution**: 368x368 input
- **Threshold**: 0.1 confidence
- **Processing**: Real-time frame analysis

### Form Metrics Calculated
| Metric | Optimal Range | Description |
|--------|---------------|-------------|
| Back Angle | 160-200° | Spine neutrality |
| Hip Hinge | 30-90° | Hip movement pattern |
| Knee Angle | 160-180° | Knee position |
| Bar Path | <30% thigh | Vertical alignment |

### Scoring Algorithm
- **Excellent**: 80-100% (Green)
- **Good**: 50-79% (Orange)
- **Needs Work**: 0-49% (Red)

Severity-based penalties applied for:
- Rounded back (most critical)
- Poor hip hinge
- Incorrect knee position
- Bar drift from body

---

## 🔧 Configuration

### Adjust Detection Sensitivity
Edit `Backend Files/pose_analyzer.py`:
```python
self.threshold = 0.1  # Lower = more lenient (0.0-1.0)
```

### Modify Form Thresholds
```python
self.BACK_ANGLE_MIN = 160  # Minimum back angle
self.BACK_ANGLE_MAX = 200  # Maximum back angle
self.HIP_ANGLE_MIN = 30    # Minimum hip hinge
self.HIP_ANGLE_MAX = 90    # Maximum hip hinge
```

### Change Server Port
Edit `Backend Files/app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

---

## 🎓 Training Custom Models

### Prepare Training Data
1. Collect 10+ images of good deadlift form
2. Collect 10+ images of bad deadlift form
3. Place in `Training Data/good/` and `Training Data/bad/`

### Train the Model
```bash
cd "Backend Files"
python train_model.py
```

### Outputs
- `trained_model_ridge.pkl` - Ridge regression
- `trained_model_logistic.pkl` - Logistic regression
- `trained_model_random_forest.pkl` - Random forest

The app automatically loads trained models if available.

---

## 🐛 Troubleshooting

### Model Files Not Found
**Error**: "Model files not found"  
**Solution**: Download OpenPose model files (see Installation step 3)

### No Keypoints Detected
**Error**: "Could not detect enough keypoints"  
**Solution**: 
- Ensure full body is visible
- Improve lighting conditions
- Check camera angle (side view best)
- Wear contrasting clothing

### Port Already in Use
**Error**: "Address already in use"  
**Solution**: 
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9
# Or change port in app.py
```

### Video Not Displaying
**Issue**: Analyzed video shows black screen  
**Solution**: 
- Check browser console for errors
- Ensure video codec is supported (H.264)
- Try different browser (Chrome recommended)

---

## 📈 Performance

### Processing Speed
- **Images**: 2-5 seconds
- **Videos**: 30-60 seconds per minute of footage
- **Real-time**: ~10 FPS on modern hardware

### Requirements
- **Minimum**: Intel i5, 8GB RAM
- **Recommended**: Intel i7, 16GB RAM, GPU
- **Storage**: 3GB (includes model files)

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute
- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🧪 Add test cases
- 🎨 Enhance UI/UX

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### OpenPose Usage
OpenPose models are used for academic/research purposes. For commercial use, contact CMU for licensing.

**Citation:**
```bibtex
@inproceedings{cao2017realtime,
  author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
  booktitle = {CVPR},
  title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  year = {2017}
}
```
</div>

### Skills Demonstrated
- ✅ Computer Vision & Deep Learning
- ✅ Full-Stack Web Development
- ✅ Real-time Video Processing
- ✅ Machine Learning Model Training
- ✅ RESTful API Design
- ✅ Modern UI/UX Design
- ✅ Data Visualization
- ✅ Technical Documentation

---

## 🌟 Acknowledgments

- **CMU Perceptual Computing Lab** - OpenPose model
- **Flask Team** - Web framework
- **OpenCV Contributors** - Computer vision library
- **scikit-learn Team** - Machine learning tools

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star!**

**Made with ❤️ by V NAVNEETH**

© 2024 V NAVNEETH. All Rights Reserved.

</div>
