import os
import logging
from pathlib import Path

# Get logger, but ensure basicConfig is called if not already done
logger = logging.getLogger(__name__)
if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Load .env file only for NetPIE configuration
try:
    from dotenv import load_dotenv
    # Try to load .env file from current directory
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loaded .env file from: {env_path.absolute()}")
    else:
        # Try loading without explicit path (dotenv will search)
        load_dotenv()
        if os.environ.get('NETPIE_ENABLED'):
            logger.info("Loaded .env file (found via dotenv search)")
        else:
            logger.warning(f".env file not found at: {env_path.absolute()}")
except ImportError:
    # python-dotenv not installed, skip .env loading
    logger.warning("python-dotenv not installed. .env file will not be loaded.")
except Exception as e:
    logger.error(f"Error loading .env file: {e}")

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
    
    # NetPIE MQTT integration (optional)
    _netpie_enabled_raw = os.environ.get('NETPIE_ENABLED', 'False')
    NETPIE_ENABLED = _netpie_enabled_raw.lower().strip() == 'true'
    
    _app_id_raw = os.environ.get('NETPIE_APP_ID')
    NETPIE_APP_ID = _app_id_raw.strip() if _app_id_raw else None
    
    _app_key_raw = os.environ.get('NETPIE_APP_KEY')
    NETPIE_APP_KEY = _app_key_raw.strip() if _app_key_raw else None
    
    _app_secret_raw = os.environ.get('NETPIE_APP_SECRET')
    NETPIE_APP_SECRET = _app_secret_raw.strip() if _app_secret_raw else None
    
    # Debug logging (only log if enabled to avoid exposing secrets)
    if NETPIE_ENABLED:
        logger.info(f"NetPIE enabled: {NETPIE_ENABLED}")
        logger.info(f"NetPIE APP_ID present: {NETPIE_APP_ID is not None} (length: {len(NETPIE_APP_ID) if NETPIE_APP_ID else 0})")
        logger.info(f"NetPIE APP_KEY present: {NETPIE_APP_KEY is not None} (length: {len(NETPIE_APP_KEY) if NETPIE_APP_KEY else 0})")
        logger.info(f"NetPIE APP_SECRET present: {NETPIE_APP_SECRET is not None} (length: {len(NETPIE_APP_SECRET) if NETPIE_APP_SECRET else 0})")
        if not all([NETPIE_APP_ID, NETPIE_APP_KEY, NETPIE_APP_SECRET]):
            logger.warning("NetPIE is enabled but one or more credentials are missing!")
    else:
        logger.debug(f"NetPIE disabled (NETPIE_ENABLED={_netpie_enabled_raw})")

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

