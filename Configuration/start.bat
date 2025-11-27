@echo off
echo ======================================
echo DeadLift Pro - Starting Application
echo ======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Check if required files exist
if not exist "app.py" (
    echo [ERROR] app.py not found. Please ensure all files are in the correct directory.
    pause
    exit /b 1
)

if not exist "pose\mpi" (
    echo [WARNING] pose\mpi folder not found.
    echo Please download OpenPose model files:
    echo   - pose_deploy_linevec_faster_4_stages.prototxt
    echo   - pose_iter_160000.caffemodel
    echo Place them in: pose\mpi\
    echo.
    pause
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
    echo.
)

REM Activate virtual environment
echo [SETUP] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo.
echo [SETUP] Installing dependencies...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo [OK] Dependencies installed
echo.

REM Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs
if not exist "good" mkdir good
if not exist "bad" mkdir bad
if not exist "templates" mkdir templates
echo [OK] Directories created
echo.

REM Start the application
echo ======================================
echo [START] Starting DeadLift Pro...
echo ======================================
echo.
echo Application will be available at:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py