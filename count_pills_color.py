import cv2
import json
import argparse
import numpy as np
import time
from pathlib import Path
from collections import deque

# Hardware communication
try:
    import hardware
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("[WARNING] hardware.py not found, hardware control disabled")

# Flask state integration
try:
    import app_state
    APP_STATE_AVAILABLE = True
except ImportError:
    APP_STATE_AVAILABLE = False

# Core functions
def load_quad(json_path, warp_w, warp_h):
    with open(json_path, "r") as f:
        pts = np.array(json.load(f)["points"], dtype=np.float32)
    dst = np.array([[0,0],[warp_w-1,0],[warp_w-1,warp_h-1],[0,warp_h-1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(pts, dst)

def warp_topdown(bgr, M, size):
    return cv2.warpPerspective(bgr, M, size)

def build_blob_params(cfg):
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = cfg["min_thr"]
    p.maxThreshold = cfg["max_thr"]
    p.thresholdStep = cfg["thr_step"]
    p.filterByArea = True
    p.minArea = float(cfg["min_area"])
    p.maxArea = float(cfg["max_area"])
    p.filterByCircularity = True
    p.minCircularity = float(cfg["min_circ"])
    p.filterByConvexity = True
    p.minConvexity = float(cfg["min_conv"])
    p.filterByInertia = True
    p.minInertiaRatio = float(cfg["min_inertia"])
    p.filterByColor = False
    return cv2.SimpleBlobDetector_create(p)

def apply_gamma(img, gamma):
    if abs(gamma - 1.0) < 1e-3:
        return img
    inv = 1.0 / max(1e-6, gamma)
    table = (np.clip(((np.arange(256)/255.0) ** inv) * 255.0, 0, 255)).astype(np.uint8)
    return cv2.LUT(img, table)

def apply_wb_gains(bgr, gains):
    if gains is None:
        return bgr
    gB, gG, gR = gains
    out = bgr.astype(np.float32)
    out[...,0] *= gB
    out[...,1] *= gG
    out[...,2] *= gR
    return np.clip(out, 0, 255).astype(np.uint8)

def center_zoom(bgr, zoom):
    if zoom <= 1.0 + 1e-6:
        return bgr
    h, w = bgr.shape[:2]
    nh, nw = int(h/zoom), int(w/zoom)
    y1 = max(0, (h - nh)//2)
    x1 = max(0, (w - nw)//2)
    crop = bgr[y1:y1+nh, x1:x1+nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

def preprocess_frame(frame_bgr, pre):
    img = center_zoom(frame_bgr, pre["zoom"])
    if abs(pre["rotate_deg"]) > 1e-3:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), pre["rotate_deg"], 1.0)
        img = cv2.warpAffine(img, M, (w, h))
    img = cv2.convertScaleAbs(img, alpha=pre["alpha"], beta=pre["beta"])
    img = apply_gamma(img, pre["gamma"])
    img = apply_wb_gains(img, pre["wb"]) if pre["wb"] else img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=pre["clahe_clip"], tileGridSize=(pre["clahe_tile"], pre["clahe_tile"]))
    gray = clahe.apply(gray)
    if pre["blur"] > 0:
        k = pre["blur"] + (1 - pre["blur"] % 2)
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return img, gray

def create_color_mask(warped_bgr, target_color_hsv, hue_tolerance=10, sat_range=(50, 255), val_range=(50, 255)):
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    h_min = max(0, target_color_hsv[0] - hue_tolerance)
    h_max = min(180, target_color_hsv[0] + hue_tolerance)
    lower = np.array([h_min, sat_range[0], val_range[0]], dtype=np.uint8)
    upper = np.array([h_max, sat_range[1], val_range[1]], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)

def detect_with_color_mask(warped_bgr, gray_w, detector, cfg):
    blob_cfg = cfg.get("blob", {})
    
    if not blob_cfg.get("use_color_filter", False):
        kps = detector.detect(gray_w)
        return kps
    
    target_color_hsv = blob_cfg.get("target_color_hsv", [19, 124, 101])
    hue_tolerance = blob_cfg.get("hue_tolerance", 18)
    sat_range = tuple(blob_cfg.get("sat_range", [90, 255]))
    val_range = tuple(blob_cfg.get("val_range", [35, 255]))
    
    color_mask = create_color_mask(warped_bgr, target_color_hsv, hue_tolerance, sat_range, val_range)
    masked_gray = cv2.bitwise_and(gray_w, gray_w, mask=color_mask)
    kps = detector.detect(masked_gray)
    return kps

class TemporalPillTracker:
    def __init__(self, max_frames=4, cluster_distance=36.0, min_detection_ratio=0.320):
        self.max_frames = max(2, max_frames) if max_frames > 0 else 2
        self.cluster_distance = cluster_distance
        self.min_detection_ratio = min_detection_ratio
        self.detection_history = deque(maxlen=self.max_frames)
        self.frame_counter = 0
    
    def add_frame(self, keypoints):
        points = np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints], dtype=np.float32)
        self.detection_history.append((self.frame_counter, points))
        self.frame_counter += 1
    
    def get_stable_count(self):
        if len(self.detection_history) < 2:
            if self.detection_history:
                return len(self.detection_history[-1][1])
            return 0
        
        all_points = []
        for frame_idx, points in self.detection_history:
            for pt in points:
                all_points.append((pt[0], pt[1], frame_idx))
        
        if not all_points:
            return 0
        
        points_array = np.array([(p[0], p[1]) for p in all_points], dtype=np.float32)
        frame_indices = np.array([p[2] for p in all_points], dtype=np.int32)
        clusters = self._cluster_points(points_array, frame_indices)
        
        min_frames_required = max(1, int(len(self.detection_history) * self.min_detection_ratio))
        stable_count = 0
        
        for cluster_frames in clusters:
            if len(cluster_frames) >= min_frames_required:
                stable_count += 1
        
        return stable_count
    
    def _cluster_points(self, points_array, frame_indices):
        n = len(points_array)
        if n == 0:
            return []
        
        visited = np.zeros(n, dtype=bool)
        clusters = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            cluster_indices = [i]
            visited[i] = True
            
            for j in range(i + 1, n):
                if visited[j]:
                    continue
                
                dist = np.linalg.norm(points_array[i] - points_array[j])
                if dist < self.cluster_distance:
                    cluster_indices.append(j)
                    visited[j] = True
            
            cluster_frames = set(frame_indices[cluster_indices])
            clusters.append(cluster_frames)
        
        return clusters
    
    def reset(self):
        self.detection_history.clear()
        self.frame_counter = 0

# Default configuration (optimized parameters)
DEFAULTS = {
    "warp": {"quad": "", "width": 600, "height": 300},
    "preprocess": {
        "zoom": 2.5, "rotate_deg": 0.0,
        "alpha": 1.0, "beta": 0.0,
        "gamma": 1.0,
        "wb": [1.0, 1.0, 1.0],
        "clahe_clip": 2.0, "clahe_tile": 8,
        "blur": 5
    },
    "blob": {
        "min_area": 300, "max_area": 5000,
        "min_circ": 0.6, "min_conv": 0.7, "min_inertia": 0.2,
        "min_thr": 10, "max_thr": 220, "thr_step": 10,
        "use_color_filter": True,
        "target_color_hsv": [19, 124, 101],
        "hue_tolerance": 18,
        "sat_range": [90, 255],
        "val_range": [35, 255]
    },
    "temporal": {
        "max_frames": 4,
        "cluster_distance": 36.0,
        "min_detection_ratio": 0.320
    }
}

def load_settings(path):
    cfg = json.loads(json.dumps(DEFAULTS))
    if path and Path(path).exists():
        user = json.loads(Path(path).read_text())
        for k in user:
            if isinstance(user[k], dict):
                cfg[k].update(user[k])
            else:
                cfg[k] = user[k]
    return cfg

def _set_and_confirm(cap, prop, value, retries=3, delay=0.1):
    for _ in range(retries):
        cap.set(prop, value)
        time.sleep(delay)
        got = cap.get(prop)
        if abs(float(got) - float(value)) < 1.5:
            return True
    return False

def set_cam_props(cap, width=1280, height=720, fps=30):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Manual mode with retry confirmation (same as vid2.py)
    _set_and_confirm(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    _set_and_confirm(cap, cv2.CAP_PROP_AUTO_WB, 0)
    _set_and_confirm(cap, cv2.CAP_PROP_WB_TEMPERATURE, 4500)

def apply_current_settings(cap, exp, bright):
    cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, bright)

def count_pills(video_source, cfg, show_vis=False, enable_hardware=False):
    warp_w = cfg["warp"]["width"]
    warp_h = cfg["warp"]["height"]
    M = None
    if cfg["warp"]["quad"]:
        M = load_quad(cfg["warp"]["quad"], warp_w, warp_h)
    
    detector = build_blob_params(cfg["blob"])
    
    temporal_params = cfg.get("temporal", {})
    tracker = TemporalPillTracker(
        max_frames=temporal_params.get("max_frames", 4),
        cluster_distance=temporal_params.get("cluster_distance", 36.0),
        min_detection_ratio=temporal_params.get("min_detection_ratio", 0.320)
    )
    
    # Open video source (file or webcam)
    if isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source, cv2.CAP_ANY)
        is_webcam = True
    else:
        # File path
        cap = cv2.VideoCapture(str(video_source))
        is_webcam = False
        # Disable zoom for video files
        cfg["preprocess"]["zoom"] = 1.0
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source: {video_source}")
        return False
    
    # Set camera properties for webcam (same as vid2.py)
    exposure_val = -10
    brightness_val = 90
    if is_webcam:
        time.sleep(0.2)  # Give camera time to initialize (same as vid2.py)
        set_cam_props(cap)
        apply_current_settings(cap, exposure_val, brightness_val)
    
    ever_reached_8_plus = False
    previous_status = None  # Track status changes for hardware commands
    last_stable_count = 0  # Track last count for final state update
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_webcam:
                    continue  # Webcam may have temporary frame drops
                break  # Video file ended
            
            # Preprocess
            color, gray = preprocess_frame(frame, cfg["preprocess"])
            warped = warp_topdown(color, M, (warp_w, warp_h)) if M is not None else cv2.resize(color, (warp_w, warp_h))
            gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=cfg["preprocess"]["clahe_clip"], tileGridSize=(cfg["preprocess"]["clahe_tile"], cfg["preprocess"]["clahe_tile"]))
            gray_w = clahe.apply(gray_w)
            if cfg["preprocess"]["blur"] > 0:
                k = cfg["preprocess"]["blur"] + (1 - cfg["preprocess"]["blur"] % 2)
                gray_w = cv2.GaussianBlur(gray_w, (k, k), 0)
            
            # Detect pills
            kps = detect_with_color_mask(warped, gray_w, detector, cfg)
            tracker.add_frame(kps)
            stable_count = tracker.get_stable_count()
            last_stable_count = stable_count  # Track for final update
            
            # Check if threshold reached
            current_status = "OK" if stable_count >= 8 else "DEFECT"
            if stable_count >= 8:
                ever_reached_8_plus = True
            
            # Update app state with cycle-based logic (for Flask dashboard)
            cycle_finalized_verdict = None
            if APP_STATE_AVAILABLE:
                # Create annotated frame for capture (with detection overlay)
                capture_frame = None
                if show_vis:
                    # Use the visualization frame if available
                    vis_copy = warped.copy()
                    vis_copy = cv2.drawKeypoints(vis_copy, kps, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    count_color = (0, 255, 0) if stable_count >= 8 else (0, 0, 255)
                    cv2.putText(vis_copy, f"Pills: {stable_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, count_color, 2)
                    capture_frame = vis_copy
                else:
                    # Use warped frame with basic annotation
                    capture_frame = warped.copy()
                    capture_frame = cv2.drawKeypoints(capture_frame, kps, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    count_color = (0, 255, 0) if stable_count >= 8 else (0, 0, 255)
                    cv2.putText(capture_frame, f"Count: {stable_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, count_color, 2)
                
                # Update with cycle logic (pass frame for image capture)
                # Returns verdict ("OK", "DEFECT", or None) when cycle finalizes (count == 0)
                cycle_finalized_verdict = app_state.update_with_cycle_logic(stable_count, capture_frame)
            
            # Send hardware command when cycle finalizes (count returns to 0)
            # This uses cycle-based logic: only highest count matters
            if enable_hardware and HARDWARE_AVAILABLE:
                if cycle_finalized_verdict is not None:
                    # Cycle just finalized - send command based on highest count in cycle
                    hardware.send_command(cycle_finalized_verdict)
                    print(f"[HARDWARE] Cycle finalized: {cycle_finalized_verdict} (based on highest count)")
                    previous_status = cycle_finalized_verdict
                elif stable_count > 0:
                    # Update previous_status for tracking (but don't send command yet)
                    previous_status = current_status
            
            # Visualization
            if show_vis:
                vis = warped.copy()
                vis = cv2.drawKeypoints(vis, kps, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                count_color = (0, 255, 0) if stable_count >= 8 else (0, 0, 255)
                count_text = f"Pills: {stable_count}"
                cv2.putText(vis, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, count_color, 2)
                
                # Use state's last_result if available (matches Flask dashboard), otherwise use legacy logic
                if APP_STATE_AVAILABLE:
                    state_dict = app_state.get_state_dict()
                    status_text = state_dict.get('last_result', 'WAITING')
                    # Map status to colors
                    if status_text == "OK":
                        status_color = (0, 255, 0)  # Green
                    elif status_text == "DEFECT":
                        status_color = (0, 0, 255)  # Red
                    elif status_text == "WAITING":
                        status_color = (255, 165, 0)  # Orange
                    else:
                        status_color = (255, 255, 255)  # White
                else:
                    # Legacy behavior for when app_state is not available
                    status_text = "OK" if ever_reached_8_plus else "Checking..."
                    status_color = (0, 255, 0) if ever_reached_8_plus else (255, 255, 255)
                cv2.putText(vis, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Show camera settings if webcam
                if is_webcam:
                    settings_text = f"Exp={exposure_val} | Bright={brightness_val} | [w/s]=exp | [a/d]=bright | [q]=quit"
                    cv2.putText(vis, settings_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Pill Counter", vis)
                key = cv2.waitKey(1) & 0xFF
                
                # Camera settings adjustment (only for webcam)
                if is_webcam:
                    if key == ord('w'):
                        exposure_val += 1
                        apply_current_settings(cap, exposure_val, brightness_val)
                        print(f"Exposure -> {exposure_val}")
                    elif key == ord('s'):
                        exposure_val -= 1
                        apply_current_settings(cap, exposure_val, brightness_val)
                        print(f"Exposure -> {exposure_val}")
                    elif key == ord('a'):
                        brightness_val -= 10
                        apply_current_settings(cap, exposure_val, brightness_val)
                        print(f"Brightness -> {brightness_val}")
                    elif key == ord('d'):
                        brightness_val += 10
                        apply_current_settings(cap, exposure_val, brightness_val)
                        print(f"Brightness -> {brightness_val}")
                
                if key in (27, ord('q')):
                    break
    
    finally:
        cap.release()
        if show_vis:
            cv2.destroyAllWindows()
    
    # Send final hardware command (for video files or final status for webcam)
    if enable_hardware and HARDWARE_AVAILABLE:
        final_status = "OK" if ever_reached_8_plus else "DEFECT"
        if not is_webcam or previous_status != final_status:
            hardware.send_command(final_status)
    
    # Finalize any remaining cycle (for Flask dashboard)
    if APP_STATE_AVAILABLE:
        app_state.update_with_cycle_logic(0, None)
    
    return ever_reached_8_plus

def main():
    ap = argparse.ArgumentParser(description="Production pill counter with temporal tracking and HSV color matching")
    ap.add_argument("source", nargs="?", default="0", 
                    help="Video file path or webcam index (default: 0, use 1 for second camera like vid2.py)")
    ap.add_argument("--settings", default="", help="JSON settings file (optional)")
    ap.add_argument("--show", action="store_true", help="Show visualization window")
    ap.add_argument("--target-color-hsv", nargs=3, type=int, default=None,
                    help="Target color in HSV format (e.g., 19 124 101)")
    ap.add_argument("--hue-tolerance", type=int, default=None, help="Hue tolerance (default: 18)")
    ap.add_argument("--sat-range-min", type=int, default=None, help="Saturation range minimum (default: 90)")
    ap.add_argument("--val-range-min", type=int, default=None, help="Value range minimum (default: 35)")
    ap.add_argument("--hardware", action="store_true", help="Enable hardware control (send commands to actuators)")
    
    args = ap.parse_args()
    
    # Load settings
    cfg = load_settings(args.settings)
    
    # Override color parameters if provided
    if args.target_color_hsv:
        cfg["blob"]["target_color_hsv"] = list(args.target_color_hsv)
    if args.hue_tolerance is not None:
        cfg["blob"]["hue_tolerance"] = args.hue_tolerance
    if args.sat_range_min is not None:
        cfg["blob"]["sat_range"][0] = args.sat_range_min
    if args.val_range_min is not None:
        cfg["blob"]["val_range"][0] = args.val_range_min
    
    # Determine if source is webcam or video file
    try:
        video_source = int(args.source)
        is_webcam = True
    except ValueError:
        video_source = Path(args.source)
        is_webcam = False
        if not video_source.exists():
            print(f"ERROR: Video file not found: {video_source}")
            return
    
    # Count pills
    result = count_pills(video_source, cfg, show_vis=args.show, enable_hardware=args.hardware)
    
    # Print result
    if result:
        print("OK")
    else:
        print("Defect")

if __name__ == "__main__":
    main()

