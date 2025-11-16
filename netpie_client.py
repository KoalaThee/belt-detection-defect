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
        
        # Set callbacks
        _client.on_connect = _on_connect
        _client.on_disconnect = _on_disconnect
        _client.on_publish = _on_publish
        
        # Connect to broker
        logger.info(f"Connecting to NetPIE broker: {broker}:{port}")
        try:
            _client.connect(broker, port, 60)
            _client.loop_start()
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
                return True
            time.sleep(0.5)
        
        if not _connected:
            logger.warning("Connection to NetPIE timed out. Check credentials and network.")
            logger.warning("The connection attempt was made but no response received.")
            logger.warning("This could indicate:")
            logger.warning("  - Firewall blocking MQTT port 1883")
            logger.warning("  - Incorrect credentials")
            logger.warning("  - Network connectivity issues")
            return False
        
        return _connected
        
    except Exception as e:
        logger.error(f"Failed to initialize NetPIE: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return False

def _on_connect(client, userdata, flags, rc):
    """Callback when MQTT client connects."""
    global _connected
    if rc == 0:
        _connected = True
        logger.info("Connected to NetPIE successfully")
    else:
        _connected = False
        logger.error(f"Failed to connect to NetPIE. Return code: {rc}")

def _on_disconnect(client, userdata, rc):
    """Callback when MQTT client disconnects."""
    global _connected
    _connected = False
    logger.warning("Disconnected from NetPIE")

def _on_publish(client, userdata, mid):
    """Callback when message is published."""
    logger.info(f"Message published to NetPIE successfully (mid: {mid})")

def publish_defect_data(state_dict: dict):
    """
    Publish defect detection data to NetPIE.
    
    Args:
        state_dict: Dictionary containing detection state (from app_state.get_state_dict())
    """
    global _client, _connected
    
    # Check prerequisites with detailed logging
    if not MQTT_AVAILABLE:
        logger.debug("NetPIE publish skipped: MQTT not available")
        return False
    
    if _client is None:
        logger.debug("NetPIE publish skipped: Client is None")
        return False
    
    if not _connected:
        logger.debug("NetPIE publish skipped: Not connected (connection status: False)")
        return False
    
    logger.debug(f"NetPIE publish: Client exists, connected={_connected}, attempting to publish...")
    
    try:
        # Prepare data payload for NetPIE widget
        payload = {
            "status": state_dict.get("last_result", "WAITING"),
            "last_count": state_dict.get("last_count", 0),
            "total_ok": state_dict.get("total_ok", 0),
            "total_defect": state_dict.get("total_defect", 0),
            "highest_count": state_dict.get("highest_count", 0),
            "last_updated": state_dict.get("last_updated", ""),
            "detection_running": state_dict.get("detection_running", False)
        }
        
        # Convert to JSON
        json_payload = json.dumps(payload)
        
        # Publish to NetPIE topic
        # Try both @msg/ and @shadow/ prefixes - NetPIE uses @shadow/ for shadow data
        topics_to_try = [
            "@shadow/data/update",  # Standard NetPIE shadow update topic
            "@msg/defect_detection",  # Custom message topic
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
                    logger.info(f"Published defect data to NetPIE - Status: {payload['status']}, Count: {payload['last_count']}, Topic: {topic}")
                    logger.debug(f"Full payload: {payload}")
                    success = True
                    # Don't break - try both topics
                else:
                    logger.warning(f"Failed to publish to NetPIE topic {topic}. Return code: {result.rc}")
            except Exception as topic_error:
                logger.warning(f"Error publishing to topic {topic}: {topic_error}")
        
        return success
            
    except Exception as e:
        logger.error(f"Error publishing to NetPIE: {e}")
        return False

def is_connected():
    """Check if NetPIE client is connected."""
    return _connected and _client is not None

def disconnect():
    """Disconnect from NetPIE."""
    global _client, _connected
    if _client:
        try:
            _client.loop_stop()
            _client.disconnect()
            _connected = False
            logger.info("Disconnected from NetPIE")
        except Exception as e:
            logger.error(f"Error disconnecting from NetPIE: {e}")

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

