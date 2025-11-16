# netpie_client.py
"""
NetPIE MQTT client for publishing defect detection data.
Publishes detection results to NetPIE platform for widget display.
"""
import json
import logging
import threading
import time
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import paho.mqtt.client as mqtt

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
    MQTT_ERR_SUCCESS = mqtt.MQTT_ERR_SUCCESS
except ImportError:
    MQTT_AVAILABLE = False
    mqtt = None  # Set to None when not available
    MQTT_ERR_SUCCESS = 0  # Default value (MQTT_ERR_SUCCESS is typically 0)
    print("[WARNING] paho-mqtt not installed. Install with: pip install paho-mqtt")

logger = logging.getLogger(__name__)

# NetPIE Configuration (set via environment variables or modify here)
APP_ID = None  # Set via NETPIE_APP_ID environment variable
APP_KEY = None  # Set via NETPIE_APP_KEY environment variable
APP_SECRET = None  # Set via NETPIE_APP_SECRET environment variable

# MQTT client instance
_client: Optional["mqtt.Client"] = None
_connected = False
_lock = threading.Lock()
_reconnect_thread: Optional[threading.Thread] = None
_reconnect_enabled = True
_last_connection_attempt = 0
_reconnect_delay = 1  # seconds between reconnection attempts
_window_open = True  # Track if OpenCV window is open (default True to keep connection alive)
_heartbeat_timer: Optional[threading.Timer] = None
_heartbeat_interval = 1.0  # seconds between heartbeat updates
_last_publish_time = 0
_publish_throttle = 0.1  # Minimum seconds between real-time publishes (100ms)

def initialize_netpie(app_id: str, app_key: str, app_secret: str):
    """
    Initialize NetPIE connection with credentials.
    
    Args:
        app_id: NetPIE Application ID
        app_key: NetPIE Application Key
        app_secret: NetPIE Application Secret
    """
    global APP_ID, APP_KEY, APP_SECRET, _client, _connected
    
    if not MQTT_AVAILABLE:
        logger.warning("paho-mqtt not available. NetPIE integration disabled.")
        return False
    
    # Check if already connected with same credentials
    if _connected and _client is not None:
        if APP_ID == app_id and APP_KEY == app_key and APP_SECRET == app_secret:
            logger.debug("NetPIE already connected with same credentials, skipping re-initialization")
            return True
        else:
            logger.warning("NetPIE already connected with different credentials, disconnecting first")
            disconnect()
    
    APP_ID = app_id
    APP_KEY = app_key
    APP_SECRET = app_secret
    
    try:
        # NetPIE uses standard broker, not token-based subdomain
        # According to NetPIE docs: Host=mqtt.netpie.io, Username=Token, ClientID=ClientID, Password=Secret
        broker = "mqtt.netpie.io"
        port = 1883
        
        # Test DNS resolution
        logger.info(f"Testing DNS resolution for broker: {broker}")
        import socket
        try:
            ip_address = socket.gethostbyname(broker)
            logger.info(f"DNS resolution successful for {broker} -> {ip_address}")
        except socket.gaierror as dns_error:
            logger.error(f"DNS resolution failed for {broker}: {dns_error}")
            logger.error("Cannot resolve NetPIE broker. Check your internet connection and DNS settings.")
            return False
        except Exception as dns_error:
            logger.warning(f"DNS test failed: {dns_error}")
        
        # NetPIE MQTT authentication:
        # - Client ID: Client ID (UUID format) from portal
        # - Username: Token (alphanumeric) from portal  
        # - Password: Secret (alphanumeric) from portal
        logger.info(f"Connecting with Client ID: ...{APP_KEY[-8:] if len(APP_KEY) > 8 else APP_KEY}")
        logger.info(f"Connecting with Username (Token): ...{APP_ID[-8:] if len(APP_ID) > 8 else APP_ID}")
        
        _client = mqtt.Client(client_id=APP_KEY)  # Client ID = Client ID from portal
        _client.username_pw_set(APP_ID, APP_SECRET)  # Username = Token, Password = Secret
        
        # Configure keepalive and connection settings
        _client._keepalive = 360  # Keepalive interval in seconds
        _client._clean_session = True
        
        # Set Last Will and Testament (LWT) to notify when device goes offline
        lwt_payload = json.dumps({"status": "OFFLINE", "timestamp": int(time.time() * 1000)})
        _client.will_set("@shadow/data/update", lwt_payload, qos=1, retain=False)
        
        # Set callbacks
        _client.on_connect = _on_connect
        _client.on_disconnect = _on_disconnect
        _client.on_publish = _on_publish
        _client.on_log = _on_log  # Optional: for debugging
        
        # Connect to broker
        logger.info(f"Connecting to NetPIE broker: {broker}:{port}")
        try:
            _client.connect(broker, port, keepalive=360)  # 60 second keepalive
            _client.loop_start()  # Start network loop in background thread
        except socket.gaierror as e:
            logger.error(f"DNS resolution error during connection: {e}")
            logger.error("Cannot resolve hostname. Check your internet connection and DNS settings.")
            return False
        except Exception as conn_error:
            logger.error(f"Connection error: {conn_error}")
            logger.error("Possible causes:")
            logger.error("  - Firewall blocking port 1883 (MQTT)")
            logger.error("  - Network connectivity issues")
            logger.error("  - Incorrect broker address")
            return False
        
        # Wait for connection (up to 5 seconds)
        for _ in range(10):
            if _connected:
                logger.info(f"Successfully connected to NetPIE broker: {broker}")
                # Start connection monitoring thread
                _start_connection_monitor()
                return True
            time.sleep(0.5)
        
        if not _connected:
            logger.warning("Connection to NetPIE timed out. Check credentials and network.")
            logger.warning("The connection attempt was made but no response received.")
            logger.warning("This could indicate:")
            logger.warning("  - Firewall blocking MQTT port 1883")
            logger.warning("  - Incorrect credentials")
            logger.warning("  - Network connectivity issues")
            # Start reconnection thread even if initial connection failed
            _start_connection_monitor()
            return False
        
        return _connected
        
    except Exception as e:
        logger.error(f"Failed to initialize NetPIE: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return False

def _on_connect(client, userdata, flags, rc):
    """Callback when MQTT client connects."""
    global _connected, _last_connection_attempt
    if rc == 0:
        _connected = True
        _last_connection_attempt = time.time()
        logger.info("Connected to NetPIE successfully")
        # Publish online status
        try:
            online_payload = {"data": {"status": "ONLINE", "timestamp": int(time.time() * 1000)}}
            client.publish("@shadow/data/update", json.dumps(online_payload), qos=1)
        except Exception as e:
            logger.debug(f"Could not publish online status: {e}")
        # Start heartbeat to keep device online
        _start_heartbeat()
    else:
        _connected = False
        logger.error(f"Failed to connect to NetPIE. Return code: {rc}")

def _on_disconnect(client, userdata, rc):
    """Callback when MQTT client disconnects."""
    global _connected
    _connected = False
    
    # Stop heartbeat when disconnected
    _stop_heartbeat()
    
    if rc == 0:
        logger.info("Disconnected from NetPIE (clean disconnect)")
    else:
        logger.warning(f"Disconnected from NetPIE unexpectedly (rc={rc}). Will attempt to reconnect...")
        # Trigger reconnection attempt
        _start_connection_monitor()

def _on_publish(client, userdata, mid):
    """Callback when message is published."""
    logger.debug(f"Message published to NetPIE successfully (mid: {mid})")

def _on_log(client, userdata, level, buf):
    """Callback for MQTT logging (optional, for debugging)."""
    # Only log warnings and errors to avoid spam
    # MQTT_LOG_WARNING is typically 4, MQTT_LOG_ERR is 8
    if MQTT_AVAILABLE and hasattr(mqtt, 'MQTT_LOG_WARNING'):
        if level <= mqtt.MQTT_LOG_WARNING:
            logger.debug(f"MQTT: {buf}")
    elif level >= 4:  # Fallback: log if level >= 4 (warning/error)
        logger.debug(f"MQTT: {buf}")

def publish_defect_data(state_dict: dict, force: bool = False):
    """
    Publish defect detection data to NetPIE in real-time.
    This publishes detection status, but heartbeat keeps device ONLINE.
    
    Args:
        state_dict: Dictionary containing detection state (from app_state.get_state_dict())
        force: If True, bypass throttle and publish immediately
    """
    global _client, _connected, _last_publish_time, _publish_throttle
    
    # Check prerequisites with detailed logging
    if not MQTT_AVAILABLE:
        logger.debug("NetPIE publish skipped: MQTT not available")
        return False
    
    if _client is None:
        logger.debug("NetPIE publish skipped: Client is None")
        return False
    
    # Verify connection is actually active
    if not _connected:
        logger.debug("NetPIE publish skipped: Not connected (connection status: False)")
        return False
    
    # Double-check client connection status
    try:
        if hasattr(_client, 'is_connected') and not _client.is_connected():
            logger.debug("NetPIE publish skipped: Client reports not connected")
            _connected = False  # Update our flag
            return False
    except Exception:
        pass  # Continue if check fails
    
    # Throttle to avoid too many messages (unless forced)
    current_time = time.time()
    if not force and (current_time - _last_publish_time) < _publish_throttle:
        return True  # Skip this publish, but return success (throttled)
    
    logger.debug(f"NetPIE publish: Client exists, connected={_connected}, attempting to publish...")
    
    try:
        # Prepare data payload for NetPIE widget
        # Include device_status: ONLINE to ensure device stays online
        payload = {
            "status": state_dict.get("last_result", "WAITING"),  # Detection status (OK/DEFECT/WAITING)
            "device_status": "ONLINE",  # Always ONLINE to keep device online
            "last_count": state_dict.get("last_count", 0),
            "total_ok": state_dict.get("total_ok", 0),
            "total_defect": state_dict.get("total_defect", 0),
            "highest_count": state_dict.get("highest_count", 0),
            "last_updated": state_dict.get("last_updated", ""),
            "detection_running": state_dict.get("detection_running", False),
            "timestamp": int(time.time() * 1000)  # Add timestamp for real-time tracking
        }
        
        # Convert to JSON
        json_payload = json.dumps(payload)
        
        # Publish to NetPIE topic
        # Try both @msg/ and @shadow/ prefixes - NetPIE uses @shadow/ for shadow data
        topics_to_try = [
            "@shadow/data/update",  # Standard NetPIE shadow update topic
            "@msg/defect_detection",  # Custom message topic for real-time updates
        ]
        
        success = False
        for topic in topics_to_try:
            try:
                # For @shadow/data/update, wrap payload in "data" field
                if topic == "@shadow/data/update":
                    shadow_payload = {"data": payload}
                    json_payload_shadow = json.dumps(shadow_payload)
                    result = _client.publish(topic, json_payload_shadow, qos=1)
                else:
                    result = _client.publish(topic, json_payload, qos=1)
                
                if result.rc == MQTT_ERR_SUCCESS:
                    logger.debug(f"Published defect data to NetPIE - Status: {payload['status']}, Count: {payload['last_count']}, Topic: {topic}")
                    success = True
                    # Don't break - try both topics
                else:
                    logger.warning(f"Failed to publish to NetPIE topic {topic}. Return code: {result.rc}")
            except Exception as topic_error:
                logger.warning(f"Error publishing to topic {topic}: {topic_error}")
        
        if success:
            _last_publish_time = current_time
        
        return success
            
    except Exception as e:
        logger.error(f"Error publishing to NetPIE: {e}")
        return False

def publish_realtime_update(state_dict: dict):
    """
    Publish real-time update immediately (bypasses throttle).
    Use this for critical state changes that need immediate notification.
    
    Args:
        state_dict: Dictionary containing detection state
    """
    return publish_defect_data(state_dict, force=True)

def _send_heartbeat():
    """Send heartbeat update to NETPIE shadow to keep device online."""
    global _client, _connected, _heartbeat_timer, _heartbeat_interval, _window_open
    
    # Always check connection status before sending heartbeat
    if not _connected or _client is None:
        # Try to restart heartbeat if connection is restored
        if _window_open and APP_ID and APP_KEY and APP_SECRET:
            logger.debug("Heartbeat: Connection lost, will retry on next cycle")
        return
    
    # Check window status but don't stop heartbeat if window status is unclear
    if not _window_open:
        return
    
    try:
        # Always send ONLINE status in heartbeat to keep device online
        # Get current state if available for additional data
        try:
            import app_state
            state_dict = app_state.get_state_dict()
            heartbeat_payload = {
                "data": {
                    "status": "ONLINE",  # Always ONLINE for heartbeat to keep device online
                    "device_status": "ONLINE",  # Explicit device status
                    "heartbeat": True,
                    "timestamp": int(time.time() * 1000),
                    "last_result": state_dict.get("last_result", "WAITING"),  # Detection result
                    "last_count": state_dict.get("last_count", 0),
                    "total_defect": state_dict.get("total_defect", 0),
                    "total_ok": state_dict.get("total_ok", 0),
                    "detection_running": state_dict.get("detection_running", False)
                }
            }
        except Exception:
            # Fallback if app_state is not available
            heartbeat_payload = {
                "data": {
                    "status": "ONLINE",  # Always ONLINE for heartbeat
                    "device_status": "ONLINE",
                    "heartbeat": True,
                    "timestamp": int(time.time() * 1000)
                }
            }
        
        json_payload = json.dumps(heartbeat_payload)
        
        # Verify client is still connected before publishing
        if _client and hasattr(_client, 'is_connected'):
            if not _client.is_connected():
                logger.debug("Heartbeat: Client not connected, skipping this heartbeat")
                # Connection lost, will be handled by connection monitor
                return
        
        result = _client.publish("@shadow/data/update", json_payload, qos=1)
        
        if result.rc == MQTT_ERR_SUCCESS:
            logger.debug(f"Heartbeat sent to NetPIE - device status: ONLINE")
        else:
            logger.warning(f"Heartbeat failed (rc={result.rc}) - connection may be lost")
            # If heartbeat fails, connection might be lost - connection monitor will handle
    except Exception as e:
        logger.warning(f"Error sending heartbeat: {e}")
    finally:
        # Always schedule next heartbeat if still connected and window open
        # This ensures heartbeat continues even if one send fails
        if _connected and _window_open and _client is not None:
            try:
                _heartbeat_timer = threading.Timer(_heartbeat_interval, _send_heartbeat)
                _heartbeat_timer.daemon = True
                _heartbeat_timer.start()
            except Exception as timer_error:
                logger.warning(f"Error scheduling next heartbeat: {timer_error}")

def _start_heartbeat():
    """Start periodic heartbeat updates to keep device online."""
    global _heartbeat_timer, _heartbeat_interval, _connected, _window_open
    
    if not _connected:
        return
    
    # Stop existing heartbeat if running
    _stop_heartbeat()
    
    logger.info(f"Starting heartbeat updates every {_heartbeat_interval} seconds to keep device online")
    
    # Send first heartbeat immediately, then schedule periodic
    _send_heartbeat()

def _stop_heartbeat():
    """Stop periodic heartbeat updates."""
    global _heartbeat_timer
    
    if _heartbeat_timer:
        _heartbeat_timer.cancel()
        _heartbeat_timer = None
        logger.debug("Stopped heartbeat updates")

def set_window_status(is_open: bool):
    """
    Set the status of the OpenCV window.
    When window is closed, NETPIE will disconnect.
    
    Args:
        is_open: True if window is open, False if closed
    """
    global _window_open
    _window_open = is_open
    if not is_open:
        logger.info("OpenCV window closed - disconnecting from NetPIE")
        _stop_heartbeat()
        disconnect()
    elif is_open and _connected:
        # Restart heartbeat if window is opened and connected
        _start_heartbeat()

def is_window_open():
    """Check if OpenCV window is still open."""
    return _window_open

def _start_connection_monitor():
    """Start background thread to monitor connection and reconnect if needed."""
    global _reconnect_thread, _reconnect_enabled
    
    if not _reconnect_enabled:
        return
    
    # Don't start multiple monitor threads
    if _reconnect_thread and _reconnect_thread.is_alive():
        return
    
    def connection_monitor():
        """Monitor connection and attempt reconnection if disconnected."""
        global _connected, _client, _last_connection_attempt, _window_open
        
        while _reconnect_enabled and _window_open:
            try:
                time.sleep(10)  # Check every 10 seconds
                
                # If window is closed, stop monitoring and disconnect
                if not _window_open:
                    logger.info("Window closed - stopping connection monitor")
                    break
                
                # Check if we should attempt reconnection
                if not _connected and APP_ID and APP_KEY and APP_SECRET and _window_open:
                    current_time = time.time()
                    # Only attempt reconnection if enough time has passed since last attempt
                    if current_time - _last_connection_attempt >= _reconnect_delay:
                        logger.info("Attempting to reconnect to NetPIE...")
                        _last_connection_attempt = current_time
                        
                        try:
                            # Check if client exists and loop is running
                            if _client is None:
                                # Reinitialize client (heartbeat will start automatically on connect)
                                initialize_netpie(APP_ID, APP_KEY, APP_SECRET)
                            else:
                                # Try to reconnect existing client
                                if not _client.is_connected():
                                    logger.info("Reconnecting MQTT client...")
                                    _client.reconnect()
                                    # Wait a bit for connection
                                    time.sleep(2)
                                    if _connected:
                                        # Reconnection successful, restart heartbeat
                                        _start_heartbeat()
                                    else:
                                        # If reconnect failed, reinitialize
                                        logger.info("Reconnect failed, reinitializing...")
                                        disconnect()
                                        time.sleep(1)
                                        initialize_netpie(APP_ID, APP_KEY, APP_SECRET)
                        except Exception as reconnect_error:
                            logger.warning(f"Reconnection attempt failed: {reconnect_error}")
                            logger.info(f"Will retry in {_reconnect_delay} seconds...")
                
                # Check if connected but loop might have stopped
                elif _connected and _client is not None and _window_open:
                    try:
                        # Verify connection is still alive by checking loop status
                        if not _client._thread:
                            logger.warning("MQTT loop thread stopped, restarting...")
                            _client.loop_start()
                            time.sleep(1)
                    except Exception as loop_error:
                        logger.debug(f"Error checking loop status: {loop_error}")
                        
            except Exception as monitor_error:
                logger.error(f"Error in connection monitor: {monitor_error}")
                time.sleep(5)  # Wait before retrying monitor loop
    
    _reconnect_thread = threading.Thread(target=connection_monitor, daemon=True)
    _reconnect_thread.start()
    logger.info("Connection monitor thread started")

def is_connected():
    """Check if NetPIE client is connected."""
    global _client, _connected
    
    if not _client:
        return False
    
    # Double-check connection status
    try:
        # Check if client thinks it's connected
        if hasattr(_client, 'is_connected'):
            client_connected = _client.is_connected()
            if client_connected != _connected:
                logger.debug(f"Connection status mismatch: client={client_connected}, flag={_connected}")
                _connected = client_connected
    except Exception:
        pass
    
    return _connected and _client is not None

def disconnect():
    """Disconnect from NetPIE."""
    global _client, _connected, _reconnect_enabled, _reconnect_thread
    
    # Stop heartbeat first
    _stop_heartbeat()
    
    # Disable reconnection
    _reconnect_enabled = False
    
    if _client:
        try:
            _client.loop_stop()
            _client.disconnect()
            _connected = False
            logger.info("Disconnected from NetPIE")
        except Exception as e:
            logger.error(f"Error disconnecting from NetPIE: {e}")
    
    # Wait for monitor thread to finish (with timeout)
    if _reconnect_thread and _reconnect_thread.is_alive():
        _reconnect_thread.join(timeout=2)

def test_publish():
    """Test function to publish a sample message for debugging."""
    test_data = {
        "status": "WAITING",
        "last_count": 0,
        "total_ok": 0,
        "total_defect": 0,
        "highest_count": 0,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detection_running": True
    }
    logger.info("Testing NetPIE publish with sample data...")
    logger.info(f"Connection status: {_connected}, Client exists: {_client is not None}")
    result = publish_defect_data(test_data)
    if result:
        logger.info("✓ Test publish successful!")
    else:
        logger.warning("✗ Test publish failed - check logs above for details")
    return result

def force_publish_test():
    """Force publish a test message directly to @shadow/data/update for debugging."""
    global _client, _connected
    
    if not _connected or _client is None:
        logger.error("Cannot force publish: Not connected to NetPIE")
        return False
    
    test_payload = {
        "data": {
            "status": "TEST",
            "last_count": 99,
            "total_ok": 1,
            "total_defect": 1,
            "highest_count": 99,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detection_running": True,
            "test": True
        }
    }
    
    try:
        topic = "@shadow/data/update"
        json_payload = json.dumps(test_payload)
        logger.info(f"Force publishing test data to {topic}...")
        logger.info(f"Payload: {json_payload}")
        result = _client.publish(topic, json_payload, qos=1)
        
        if result.rc == MQTT_ERR_SUCCESS:
            logger.info(f"✓ Force publish successful! Check NetPIE shadow in a few seconds.")
            return True
        else:
            logger.error(f"✗ Force publish failed. Return code: {result.rc}")
            return False
    except Exception as e:
        logger.error(f"✗ Force publish error: {e}")
        return False

