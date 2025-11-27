#!/bin/bash

echo "======================================"
echo "DeadLift Pro - Starting Application"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if required files exist
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found. Please ensure all files are in the correct directory."
    exit 1
fi

if [ ! -d "pose/mpi" ]; then
    echo "⚠ Warning: pose/mpi folder not found."
    echo "Please download OpenPose model files:"
    echo "  - pose_deploy_linevec_faster_4_stages.prototxt"
    echo "  - pose_iter_160000.caffemodel"
    echo "Place them in: pose/mpi/"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

# Install requirements
echo ""
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create necessary directories
mkdir -p uploads outputs good bad templates
echo "✓ Directories created"
echo ""

# Start the application
echo "======================================"
echo "🚀 Starting DeadLift Pro..."
echo "======================================"
echo ""
echo "Application will be available at:"
echo "  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py