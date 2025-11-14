import os
from pathlib import Path

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Flask settings
    HOST = os.environ.get('FLASK_HOST') or '0.0.0.0'
    PORT = int(os.environ.get('FLASK_PORT') or 5000)
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Detection settings
    # VIDEO_SOURCE can be: int (camera index) or Path/str (video file path)
    _video_source_env = os.environ.get('VIDEO_SOURCE')
    if _video_source_env:
        try:
            VIDEO_SOURCE = int(_video_source_env)  # Try camera index first
        except ValueError:
            VIDEO_SOURCE = Path(_video_source_env)  # If not int, treat as file path
    else:
        VIDEO_SOURCE = 1  # Default: webcam (camera index 1)
    SETTINGS_FILE = os.environ.get('SETTINGS_FILE') or None
    SHOW_VISUALIZATION = os.environ.get('SHOW_VIS', 'False').lower() == 'true'
    ENABLE_HARDWARE = os.environ.get('ENABLE_HARDWARE', 'True').lower() == 'true'
    
    # API settings
    API_POLL_INTERVAL = 1000  # milliseconds for dashboard polling

class DevelopmentConfig(Config):
    DEBUG = True
    SHOW_VISUALIZATION = True

class ProductionConfig(Config):
    DEBUG = False
    SHOW_VISUALIZATION = False

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

