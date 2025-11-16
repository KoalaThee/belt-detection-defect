# NetPIE Code Changes Documentation

This document describes all the code changes made to integrate NetPIE MQTT functionality into the pill detection system.

## Overview

NetPIE integration allows the pill detection system to publish detection results to the NetPIE IoT platform for real-time monitoring and dashboard visualization.

---

## Files Modified/Created

### 1. `netpie_client.py` (Created/Modified)
**Purpose**: MQTT client for connecting to NetPIE and publishing detection data.

### 2. `app_flask.py` (Modified)
**Purpose**: Initialize NetPIE connection when Flask app starts.

### 3. `app_state.py` (Modified)
**Purpose**: Automatically publish detection data to NetPIE when state updates.

### 4. `config_flask.py` (Modified)
**Purpose**: Load NetPIE credentials from `.env` file.

---

## Code That Already Existed (Modified/Improved)

### 1. Data Publishing Function (`netpie_client.py`)

**Function**: `publish_defect_data(state_dict: dict)`

**What it does**: Publishes detection state data to NetPIE MQTT broker.

**Improvements made**:
- Enhanced logging (changed from DEBUG to INFO level)
- Added detailed logging showing status, count, and topic when publishing
- Better error messages
- Added debug logging for full payload

**Code location**: `netpie_client.py` lines 161-202

```python
def publish_defect_data(state_dict: dict):
    """
    Publish defect detection data to NetPIE.
    
    Args:
        state_dict: Dictionary containing detection state (from app_state.get_state_dict())
    """
    # Checks connection status
    # Prepares JSON payload
    # Publishes to topic: @msg/defect_detection
    # Returns True/False for success/failure
```

### 2. Publishing Calls (`app_state.py`)

**Already existed** - Automatic publishing happens in two places:

#### Location 1: When Detection Cycle Finalizes (Line 70-78)
```python
# Publish to NetPIE if available (when cycle finalizes)
if NETPIE_AVAILABLE:
    try:
        state_dict = asdict(_state)
        state_dict['has_image'] = _state.has_image
        netpie_client.publish_defect_data(state_dict)
    except Exception as e:
        # Don't fail if NetPIE publish fails
        pass
```
**When**: When count returns to 0 (pill has passed detection area)

#### Location 2: When Detection State Updates (Line 109-117)
```python
# Publish to NetPIE if available
if NETPIE_AVAILABLE:
    try:
        state_dict = asdict(_state)
        state_dict['has_image'] = _state.has_image
        netpie_client.publish_defect_data(state_dict)
    except Exception as e:
        # Don't fail if NetPIE publish fails
        pass
```
**When**: Every frame where pills are detected (real-time updates)

---

## Code Added/Modified

### 1. NetPIE Initialization (`netpie_client.py` & `app_flask.py`)

#### `netpie_client.py` - `initialize_netpie()` function

**Key fixes**:
- ✅ Fixed broker URL: Changed from `{token}.netpie.io` to `mqtt.netpie.io` (standard broker)
- ✅ Fixed authentication: 
  - Client ID = Client ID from portal (UUID format)
  - Username = Token from portal (alphanumeric)
  - Password = Secret from portal
- ✅ Added DNS resolution testing before connection
- ✅ Added connection state checks to prevent duplicate initializations
- ✅ Improved error handling and logging

**Code location**: `netpie_client.py` lines 38-139

```python
def initialize_netpie(app_id: str, app_key: str, app_secret: str):
    """
    Initialize NetPIE connection with credentials.
    
    Args:
        app_id: NetPIE Token (used as MQTT username)
        app_key: NetPIE Client ID (used as MQTT client_id)
        app_secret: NetPIE Secret (used as MQTT password)
    """
    # Checks if already connected
    # Tests DNS resolution
    # Creates MQTT client with correct credentials
    # Connects to mqtt.netpie.io:1883
    # Waits for connection confirmation
    # Returns True/False
```

#### `app_flask.py` - NetPIE initialization in `create_app()`

**Key additions**:
- ✅ Added Flask reloader detection to prevent multiple connections
- ✅ Added connection status check before initializing
- ✅ Enhanced logging for debugging
- ✅ Proper error handling for missing credentials

**Code location**: `app_flask.py` lines 40-82

```python
# Initialize NetPIE if enabled
# Skip initialization in Flask reloader parent process
is_reloader_parent = os.environ.get('WERKZEUG_RUN_MAIN') is None

if NETPIE_AVAILABLE and app.config.get('NETPIE_ENABLED'):
    if is_reloader_parent:
        logger.info("NetPIE: Skipping initialization in Flask reloader parent process")
    else:
        # Check if already connected
        if netpie_client.is_connected():
            logger.info("NetPIE already connected, skipping re-initialization")
        elif app_id and app_key and app_secret:
            # Initialize connection
            netpie_client.initialize_netpie(app_id, app_key, app_secret)
```

### 2. Connection Management Functions (`netpie_client.py`)

#### `is_connected()` - Check connection status
**Code location**: `netpie_client.py` lines 204-206

```python
def is_connected():
    """Check if NetPIE client is connected."""
    return _connected and _client is not None
```

#### `disconnect()` - Clean disconnect
**Code location**: `netpie_client.py` lines 208-218

```python
def disconnect():
    """Disconnect from NetPIE."""
    # Stops MQTT loop
    # Disconnects from broker
    # Resets connection state
```

#### `test_publish()` - Test function for debugging
**Code location**: `netpie_client.py` lines 220-231

```python
def test_publish():
    """Test function to publish a sample message for debugging."""
    # Creates test data
    # Publishes to NetPIE
    # Returns success/failure
```

### 3. MQTT Callback Functions (`netpie_client.py`)

#### `_on_connect()` - Connection callback
**Code location**: `netpie_client.py` lines 141-149

```python
def _on_connect(client, userdata, flags, rc):
    """Callback when MQTT client connects."""
    # Sets _connected = True if rc == 0
    # Logs connection status
```

#### `_on_disconnect()` - Disconnection callback
**Code location**: `netpie_client.py` lines 151-155

```python
def _on_disconnect(client, userdata, rc):
    """Callback when MQTT client disconnects."""
    # Sets _connected = False
    # Logs disconnection
```

#### `_on_publish()` - Publish confirmation callback
**Code location**: `netpie_client.py` lines 157-159

```python
def _on_publish(client, userdata, mid):
    """Callback when message is published."""
    # Logs successful publication
```

### 4. Configuration Loading (`config_flask.py`)

**Changes**:
- ✅ Enhanced `.env` file loading with better error handling
- ✅ Added logging to show when `.env` file is loaded
- ✅ Added whitespace stripping for environment variables
- ✅ Added debug logging for credential presence (without exposing secrets)

**Code location**: `config_flask.py` lines 7-26, 53-75

```python
# Load .env file with error handling
env_path = Path('.env')
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded .env file from: {env_path.absolute()}")

# NetPIE configuration with validation
NETPIE_ENABLED = os.environ.get('NETPIE_ENABLED', 'False').lower().strip() == 'true'
NETPIE_APP_ID = os.environ.get('NETPIE_APP_ID').strip() if os.environ.get('NETPIE_APP_ID') else None
# ... etc
```

### 5. Cleanup on Shutdown (`app_flask.py`)

**Code location**: `app_flask.py` lines 283-288

```python
finally:
    # Cleanup NetPIE connection on shutdown
    if NETPIE_AVAILABLE:
        try:
            netpie_client.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting NetPIE: {e}")
```

---

## Data Published to NetPIE

### Topic
`@msg/defect_detection`

### Payload Structure (JSON)
```json
{
    "status": "OK|DEFECT|WAITING",
    "last_count": 0-15,
    "total_ok": 0,
    "total_defect": 0,
    "highest_count": 0-15,
    "last_updated": "2025-11-16 17:00:00",
    "detection_running": true/false
}
```

### Publishing Frequency
- **Real-time updates**: Every frame where pills are detected
- **Cycle finalization**: When count returns to 0 (pill has passed)

---

## Key Fixes Made

### 1. Broker URL Fix
**Problem**: Code was trying to use `{token}.netpie.io` which doesn't resolve via DNS.

**Solution**: Changed to standard broker `mqtt.netpie.io`

### 2. Authentication Fix
**Problem**: Incorrect credential mapping causing "Not authorized" errors.

**Solution**: 
- Client ID = Client ID from portal (UUID)
- Username = Token from portal (alphanumeric)
- Password = Secret from portal

### 3. Flask Reloader Fix
**Problem**: Flask debug mode creates multiple processes, causing connection/disconnection loops.

**Solution**: Skip NetPIE initialization in Flask reloader parent process using `WERKZEUG_RUN_MAIN` check.

### 4. Connection State Management
**Problem**: Multiple initialization attempts causing conflicts.

**Solution**: Added `is_connected()` check before initializing.

---

## Integration Points

### Automatic Publishing
Data is automatically published when:
1. Detection state updates (every frame with pills)
2. Detection cycle finalizes (when count returns to 0)

### No Manual Code Required
Once NetPIE is enabled in `.env` file, publishing happens automatically. No additional code changes needed in detection logic.

---

## Configuration Required

### `.env` File
```env
NETPIE_ENABLED=True
NETPIE_APP_ID=your_token_here
NETPIE_APP_KEY=your_client_id_here
NETPIE_APP_SECRET=your_secret_here
```

### Credential Mapping
- `NETPIE_APP_ID` = **Token** from NetPIE portal (alphanumeric)
- `NETPIE_APP_KEY` = **Client ID** from NetPIE portal (UUID format)
- `NETPIE_APP_SECRET` = **Secret** from NetPIE portal (alphanumeric)

---

## Dependencies

### Required Package
```bash
pip install paho-mqtt
```

### Optional (for .env file loading)
```bash
pip install python-dotenv
```

---

## Testing

### Test Connection
```python
python test_netpie_connection.py
```

### Test Environment Variables
```python
python test_env.py
```

### Test Publish Function
```python
from netpie_client import test_publish
test_publish()
```

---

## Logging

### Connection Logs
- `[INFO] Connected to NetPIE successfully`
- `[INFO] Successfully connected to NetPIE broker: mqtt.netpie.io`

### Publishing Logs
- `[INFO] Published defect data to NetPIE - Status: OK, Count: 8, Topic: @msg/defect_detection`
- `[INFO] Message published to NetPIE successfully (mid: 12345)`

### Error Logs
- `[ERROR] Failed to initialize NetPIE: ...`
- `[WARNING] Disconnected from NetPIE`

---

## Summary

### What Was Already There
- ✅ `publish_defect_data()` function (improved logging)
- ✅ Publishing calls in `app_state.py` (automatic integration)

### What Was Added/Modified
- ✅ Fixed NetPIE connection (broker URL, authentication)
- ✅ Added Flask reloader handling
- ✅ Enhanced logging throughout
- ✅ Added connection state management
- ✅ Improved error handling
- ✅ Added cleanup on shutdown
- ✅ Enhanced configuration loading

### Result
NetPIE integration is now fully functional with automatic data publishing when detection runs. No manual intervention required once configured.

