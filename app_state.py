from dataclasses import dataclass, asdict
from datetime import datetime
from threading import Lock
from typing import Optional
import cv2
import numpy as np
from pathlib import Path

@dataclass
class DetectionState:
    last_result: str = "WAITING"      # "OK", "DEFECT", "WAITING", "ERROR"
    last_count: int = 0
    total_ok: int = 0
    total_defect: int = 0
    last_updated: str = ""
    detection_running: bool = False
    error_message: Optional[str] = None
    highest_count: int = 0  # Highest count in current cycle
    current_cycle_max: int = 0  # Maximum count in current detection cycle
    has_image: bool = False  # Whether an image is available

# Separate storage for image (not in dataclass to avoid serialization issues)
_highest_count_image: Optional[bytes] = None
_image_lock = Lock()

# Thread-safe state management
_state = DetectionState()
_lock = Lock()

# Create directory for captured images
CAPTURE_DIR = Path("static/captures")
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

def update_with_cycle_logic(count: int, frame_image: Optional[np.ndarray] = None, error: Optional[str] = None):
    global _state, _highest_count_image
    with _lock:
        if error:
            _state.last_result = "ERROR"
            _state.error_message = error
            _state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return
        
        _state.last_count = int(count)
        _state.error_message = None
        
        # Reset cycle when count becomes 0 (pill has passed)
        if count == 0:
            verdict = None
            if _state.current_cycle_max > 0:
                verdict = finalize_cycle()  # Returns "OK", "DEFECT", or None
            # Reset cycle state
            _state.current_cycle_max = 0
            _state.highest_count = 0
            # Always set status to WAITING when count is 0 (ensures reset from OK/DEFECT)
            _state.last_result = "WAITING"
            _state.last_count = 0
            _state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Clear image when cycle resets
            with _image_lock:
                _highest_count_image = None
                _state.has_image = False
            # Return verdict so caller can send hardware command
            return verdict
        
        # Update cycle maximum
        if count > _state.current_cycle_max:
            _state.current_cycle_max = count
            _state.highest_count = count
            
            # Capture image if provided and this is the new highest
            if frame_image is not None:
                try:
                    # Encode image as JPEG
                    _, buffer = cv2.imencode('.jpg', frame_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    with _image_lock:
                        _highest_count_image = buffer.tobytes()
                        _state.has_image = True
                except Exception as e:
                    print(f"[WARNING] Failed to capture image: {e}")
        
        # Update current status based on count
        if count >= 8:
            _state.last_result = "OK"
        elif count >= 1:
            _state.last_result = "DEFECT"
        else:
            _state.last_result = "WAITING"
        
        _state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return None  # No cycle finalized yet

def finalize_cycle():
    global _state
    max_count = _state.current_cycle_max
    
    if max_count == 0:
        return None  # No detection in this cycle
    
    # Only highest count matters: >=8 is OK, 1-7 is DEFECT
    if max_count >= 8:
        _state.total_ok += 1
        return "OK"
    elif max_count >= 1:
        _state.total_defect += 1
        return "DEFECT"
    
    return None

def update(result: str, count: int, error: Optional[str] = None):
    update_with_cycle_logic(count, None, error)

def set_detection_running(running: bool):
    global _state
    with _lock:
        _state.detection_running = running

def get_state_dict():
    with _lock:
        state_dict = asdict(_state)
        # Add image availability flag (image itself is retrieved separately)
        state_dict['has_image'] = _state.has_image
        return state_dict

def get_highest_count_image():
    with _image_lock:
        return _highest_count_image

def reset_counters():
    global _state
    with _lock:
        _state.total_ok = 0
        _state.total_defect = 0

