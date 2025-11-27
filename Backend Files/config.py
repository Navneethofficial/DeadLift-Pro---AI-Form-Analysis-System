"""
Configuration settings for DeadLift Pro application
"""

import os

class Config:
    """Base configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = True
    
    # File upload settings
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'outputs'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}
    
    # Model settings
    MODEL_MODE = "MPI"
    PROTO_FILE = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    WEIGHTS_FILE = "pose/mpi/pose_iter_160000.caffemodel"
    
    # Pose detection settings
    INPUT_WIDTH = 368
    INPUT_HEIGHT = 368
    DETECTION_THRESHOLD = 0.1
    N_POINTS = 15
    
    # Form analysis thresholds
    BACK_ANGLE_MIN = 160
    BACK_ANGLE_MAX = 200
    HIP_ANGLE_MIN = 30
    HIP_ANGLE_MAX = 90
    KNEE_ANGLE_MIN = 160
    KNEE_ANGLE_MAX = 180
    
    # Scoring thresholds
    EXCELLENT_SCORE = 0.8
    GOOD_SCORE = 0.6
    
    # Training data paths
    GOOD_POSTURES_PATH = 'good'
    BAD_POSTURES_PATH = 'bad'
    
    # Video processing
    VIDEO_FPS = 30
    VIDEO_CODEC = 'mp4v'
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # Add production-specific settings here


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}