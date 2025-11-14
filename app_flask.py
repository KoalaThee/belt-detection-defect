import logging
import threading
import time
import argparse
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import InternalServerError

import app_state
import count_pills_color as detector
import config_flask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Global detection thread reference
_detection_thread = None
_detection_running = False

def create_app(config_name='default', video_source_override=None):
    app = Flask(__name__)
    app.config.from_object(config_flask.config[config_name])
    
    # Override video source if provided via command line
    if video_source_override is not None:
        app.config['VIDEO_SOURCE'] = video_source_override
    
    # Register routes
    register_routes(app)
    
    # Start detection thread
    @app.before_request
    def ensure_detection_thread():
        global _detection_thread, _detection_running
        
        if not _detection_running and _detection_thread is None:
            start_detection_thread(app.config)
    
    return app

def register_routes(app):
    
    @app.route('/')
    def index():
        return render_template('dashboard.html', 
                             poll_interval=app.config['API_POLL_INTERVAL'])
    
    @app.route('/api/status')
    def api_status():
        try:
            state = app_state.get_state_dict()
            return jsonify({
                'success': True,
                'data': state
            })
        except Exception as e:
            logger.error(f"Error getting status: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/health')
    def api_health():
        state = app_state.get_state_dict()
        return jsonify({
            'status': 'healthy',
            'detection_running': state['detection_running'],
            'timestamp': state['last_updated']
        })
    
    @app.route('/api/reset', methods=['POST'])
    def api_reset():
        try:
            app_state.reset_counters()
            logger.info("Counters reset via API")
            return jsonify({
                'success': True,
                'message': 'Counters reset'
            })
        except Exception as e:
            logger.error(f"Error resetting counters: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/image')
    def api_image():
        try:
            image_data = app_state.get_highest_count_image()
            
            if image_data is None:
                from flask import Response
                # Return a 1x1 transparent pixel if no image
                return Response(
                    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82',
                    mimetype='image/png'
                )
            
            from flask import Response
            return Response(
                image_data,
                mimetype='image/jpeg',
                headers={
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
            )
        except Exception as e:
            logger.error(f"Error getting image: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': 'Not found'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

def start_detection_thread(config):
    global _detection_thread, _detection_running
    
    if _detection_running:
        logger.warning("Detection thread already running")
        return
    
    def detection_worker():
        global _detection_running
        
        try:
            _detection_running = True
            app_state.set_detection_running(True)
            logger.info("Starting detection worker thread")
            
            # Load configuration
            cfg = detector.DEFAULTS
            if config.get('SETTINGS_FILE'):
                try:
                    cfg = detector.load_settings(config['SETTINGS_FILE'])
                    logger.info(f"Loaded settings from {config['SETTINGS_FILE']}")
                except Exception as e:
                    logger.warning(f"Could not load settings file: {e}")
            
            video_source = config.get('VIDEO_SOURCE', 0)
            show_vis = config.get('SHOW_VISUALIZATION', False)
            enable_hardware = config.get('ENABLE_HARDWARE', True)
            
            logger.info(f"Detection config: source={video_source}, vis={show_vis}, hardware={enable_hardware}")
            
            # Run detection loop
            detector.count_pills(
                video_source=video_source,
                cfg=cfg,
                show_vis=show_vis,
                enable_hardware=enable_hardware
            )
            
        except Exception as e:
            logger.error(f"Detection worker error: {e}", exc_info=True)
            app_state.update("ERROR", 0, error=str(e))
        finally:
            _detection_running = False
            app_state.set_detection_running(False)
            logger.info("Detection worker thread stopped")
    
    _detection_thread = threading.Thread(target=detection_worker, daemon=True)
    _detection_thread.start()
    logger.info("Detection thread started")

# Create app instance
app = None

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Flask web application for pill detection')
    parser.add_argument('--video-source', type=str, default=None,
                        help='Video source: camera index (0, 1, etc.) or video file path (e.g., simulation_vid.mp4). Default: webcam (1)')
    args = parser.parse_args()
    
    # Parse video source
    video_source = None
    if args.video_source:
        try:
            # Try to parse as int (camera index)
            video_source = int(args.video_source)
            logger.info(f"Using camera index: {video_source}")
        except ValueError:
            # If not int, treat as file path
            video_source = Path(args.video_source)
            if not video_source.exists():
                logger.error(f"ERROR: Video file not found: {video_source}")
                exit(1)
            logger.info(f"Using video file: {video_source}")
    else:
        # Default: webcam (camera index 1)
        video_source = 1
        logger.info("Using default webcam (camera index 1)")
    
    # Create app with video source override
    app = create_app(video_source_override=video_source)
    
    # Development server
    config_obj = config_flask.config['default']
    app.run(
        host=config_obj.HOST,
        port=config_obj.PORT,
        debug=config_obj.DEBUG,
        threaded=True
    )

