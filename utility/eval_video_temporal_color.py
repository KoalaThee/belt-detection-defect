# eval_video_temporal_color.py
"""
Single video evaluator with temporal tracking and HSV color matching.
Processes a video and outputs duration-based accuracy percentage.
Uses HSV color masking to filter detections by pill color.
"""
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from collections import deque

# Import core functions from count_pills_simple.py
def load_quad(json_path, warp_w, warp_h):
    with open(json_path, "r") as f:
        pts = np.array(json.load(f)["points"], dtype=np.float32)
    dst = np.array([[0,0],[warp_w-1,0],[warp_w-1,warp_h-1],[0,warp_h-1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(pts, dst)

def warp_topdown(bgr, M, size):
    return cv2.warpPerspective(bgr, M, size)

def build_blob_params(cfg):
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold  = cfg["min_thr"]
    p.maxThreshold  = cfg["max_thr"]
    p.thresholdStep = cfg["thr_step"]
    p.filterByArea = True
    p.minArea = float(cfg["min_area"]); p.maxArea = float(cfg["max_area"])
    p.filterByCircularity = True; p.minCircularity = float(cfg["min_circ"])
    p.filterByConvexity  = True; p.minConvexity   = float(cfg["min_conv"])
    p.filterByInertia    = True; p.minInertiaRatio= float(cfg["min_inertia"])
    p.filterByColor = False
    return cv2.SimpleBlobDetector_create(p)

def apply_gamma(img, gamma):
    if abs(gamma - 1.0) < 1e-3: return img
    inv = 1.0 / max(1e-6, gamma)
    table = (np.clip(((np.arange(256)/255.0) ** inv) * 255.0, 0, 255)).astype(np.uint8)
    return cv2.LUT(img, table)

def apply_wb_gains(bgr, gains):
    if gains is None: return bgr
    gB, gG, gR = gains
    out = bgr.astype(np.float32)
    out[...,0] *= gB; out[...,1] *= gG; out[...,2] *= gR
    return np.clip(out, 0, 255).astype(np.uint8)

def center_zoom(bgr, zoom):
    if zoom <= 1.0 + 1e-6: return bgr
    h, w = bgr.shape[:2]
    nh, nw = int(h/zoom), int(w/zoom)
    y1 = max(0, (h - nh)//2); x1 = max(0, (w - nw)//2)
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

# ---------- HSV Color Masking Functions ----------
def create_color_mask(warped_bgr, target_color_hsv, hue_tolerance=10, sat_range=(50, 255), val_range=(50, 255)):
    """
    Create a binary mask for specific color range in HSV space.
    
    Args:
        warped_bgr: Input BGR image
        target_color_hsv: Target color in HSV [H, S, V] (0-180, 0-255, 0-255)
        hue_tolerance: Hue tolerance in degrees (0-180)
        sat_range: Saturation range tuple (min, max) (0-255)
        val_range: Value (brightness) range tuple (min, max) (0-255)
    
    Returns:
        Binary mask (255 = match, 0 = no match)
    """
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    
    # Handle hue wrap-around (red is at both 0 and 180)
    h_min = max(0, target_color_hsv[0] - hue_tolerance)
    h_max = min(180, target_color_hsv[0] + hue_tolerance)
    
    # If hue range crosses 0/180 boundary, handle separately
    if h_min < 0 or h_max > 180:
        # For simplicity, clamp to valid range
        h_min = max(0, h_min)
        h_max = min(180, h_max)
    
    lower = np.array([h_min, sat_range[0], val_range[0]], dtype=np.uint8)
    upper = np.array([h_max, sat_range[1], val_range[1]], dtype=np.uint8)
    
    # Create mask
    mask = cv2.inRange(hsv, lower, upper)
    
    return mask

def detect_with_color_mask(warped_bgr, gray_w, detector, cfg):
    """
    Detect pills using HSV color mask + blob detector.
    
    Args:
        warped_bgr: Warped color image
        gray_w: Warped grayscale image
        detector: Blob detector
        cfg: Configuration dictionary
    
    Returns:
        keypoints: Detected keypoints
        color_mask: Color mask used (for visualization)
    """
    blob_cfg = cfg.get("blob", {})
    
    # Check if color filtering is enabled
    if not blob_cfg.get("use_color_filter", False):
        # No color filtering, use standard detection
        kps = detector.detect(gray_w)
        return kps, None
    
    # Get color parameters
    target_color_hsv = blob_cfg.get("target_color_hsv", [20, 200, 200])  # Default: orange
    hue_tolerance = blob_cfg.get("hue_tolerance", 10)
    sat_range = tuple(blob_cfg.get("sat_range", [50, 255]))
    val_range = tuple(blob_cfg.get("val_range", [50, 255]))
    
    # Create color mask
    color_mask = create_color_mask(
        warped_bgr, 
        target_color_hsv, 
        hue_tolerance, 
        sat_range, 
        val_range
    )
    
    # Apply mask to grayscale image (only detect in colored regions)
    masked_gray = cv2.bitwise_and(gray_w, gray_w, mask=color_mask)
    
    # Detect blobs on masked image
    kps = detector.detect(masked_gray)
    
    return kps, color_mask

class TemporalPillTracker:
    """Tracks pill detections across multiple frames to stabilize counts."""
    def __init__(self, max_frames=30, cluster_distance=30.0, min_detection_ratio=0.3):
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
        # Color matching parameters
        "use_color_filter": False,  # Enable/disable color filtering
        "target_color_hsv": [19, 124, 101], # [30, 200, 200],  # Target color in HSV (default: orange)
        "hue_tolerance": 30,  # Hue range: ±10 degrees
        "sat_range": [41, 255],  # Saturation range
        "val_range": [2, 255]  # Value (brightness) range
    },
    "temporal": {
        "enabled": True,
        "max_frames": 5,
        "cluster_distance": 35.19,
        "min_detection_ratio": 0.139
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

def evaluate_video_with_visualization(video_path, cfg, show_vis=True):
    """
    Evaluate a single video with temporal tracking, HSV color matching, and optional visualization.
    Uses binary scoring: checks if video ever reaches >=8 pills (not time-based).
    Also tracks second utility: duration when exactly 8 pills are detected.
    Returns binary score and statistics.
    """
    warp_w = cfg["warp"]["width"]
    warp_h = cfg["warp"]["height"]
    M = None
    if cfg["warp"]["quad"]:
        M = load_quad(cfg["warp"]["quad"], warp_w, warp_h)
    
    detector = build_blob_params(cfg["blob"])
    
    # Load temporal parameters
    temporal_params = cfg.get("temporal", {})
    max_frames = temporal_params.get("max_frames", 10)
    cluster_distance = temporal_params.get("cluster_distance", 30.0)
    min_detection_ratio = temporal_params.get("min_detection_ratio", 0.3)
    
    tracker = TemporalPillTracker(
        max_frames=max_frames,
        cluster_distance=cluster_distance,
        min_detection_ratio=min_detection_ratio
    )
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return None
    
    total_frames = 0
    count_history = []
    
    # Track if video ever reached threshold (binary check, not time-based)
    ever_reached_8_plus = False
    # Track frames with exactly 8 pills (second utility)
    frames_with_exactly_8 = 0
    
    blob_cfg = cfg.get("blob", {})
    use_color = blob_cfg.get("use_color_filter", False)
    
    print(f"Processing video: {video_path.name}")
    print(f"Temporal params: max_frames={max_frames}, cluster_dist={cluster_distance:.1f}, min_ratio={min_detection_ratio:.2f}")
    if use_color:
        target_hsv = blob_cfg.get("target_color_hsv", [20, 200, 200])
        print(f"Color filtering: ENABLED (HSV={target_hsv}, hue_tol={blob_cfg.get('hue_tolerance', 10)})")
    else:
        print(f"Color filtering: DISABLED")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        # Preprocess
        color, gray = preprocess_frame(frame, cfg["preprocess"])
        warped = warp_topdown(color, M, (warp_w, warp_h)) if M is not None else cv2.resize(color, (warp_w, warp_h))
        gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=cfg["preprocess"]["clahe_clip"], tileGridSize=(cfg["preprocess"]["clahe_tile"], cfg["preprocess"]["clahe_tile"]))
        gray_w = clahe.apply(gray_w)
        if cfg["preprocess"]["blur"] > 0:
            k = cfg["preprocess"]["blur"] + (1 - cfg["preprocess"]["blur"] % 2)
            gray_w = cv2.GaussianBlur(gray_w, (k, k), 0)
        
        # Detect with color masking
        kps, color_mask = detect_with_color_mask(warped, gray_w, detector, cfg)
        tracker.add_frame(kps)
        stable_count = tracker.get_stable_count()
        current_count = len(kps)
        
        total_frames += 1
        count_history.append(stable_count)
        
        # Check if threshold was reached (binary check, not time-based)
        if stable_count >= 8:
            ever_reached_8_plus = True
        
        # Track frames with exactly 8 pills (second utility)
        if stable_count == 8:
            frames_with_exactly_8 += 1
        
        # Visualization
        if show_vis:
            vis = warped.copy()
            
            # Draw color mask overlay (semi-transparent)
            if color_mask is not None:
                mask_colored = cv2.applyColorMap(color_mask, cv2.COLORMAP_JET)
                vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
            
            # Draw detected keypoints
            vis = cv2.drawKeypoints(vis, kps, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # Color coding based on stable count
            if stable_count >= 8:
                count_color = (0, 255, 0)  # Green - OK
            elif stable_count >= 6:
                count_color = (0, 165, 255)  # Orange - partial
            else:
                count_color = (0, 0, 255)  # Red - defect
            
            count_text = f"stable={stable_count} (current={current_count})"
            cv2.putText(vis, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, count_color, 2)
            
            # Show color filter status
            if use_color:
                color_status = f"Color filter: ON (HSV={blob_cfg.get('target_color_hsv')})"
                cv2.putText(vis, color_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show binary threshold status
            threshold_status = "Reached >=8" if ever_reached_8_plus else "Not reached >=8"
            status_color = (0, 255, 0) if ever_reached_8_plus else (0, 0, 255)
            cv2.putText(vis, threshold_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Show exactly 8 pills duration (second utility)
            exactly_8_ratio = frames_with_exactly_8 / total_frames if total_frames > 0 else 0.0
            exactly_8_text = f"Exactly 8: {frames_with_exactly_8}/{total_frames} ({exactly_8_ratio:.1%})"
            cv2.putText(vis, exactly_8_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow("Video Evaluation (Color)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                print("Evaluation stopped by user")
                break
    
    cap.release()
    if show_vis:
        cv2.destroyAllWindows()
    
    if total_frames == 0:
        print("ERROR: No frames processed")
        return None
    
    # Calculate binary score
    binary_score = 1.0 if ever_reached_8_plus else 0.0
    
    # Calculate duration ratio for diagnostic purposes
    frames_with_8_plus = sum(1 for c in count_history if c >= 8)
    duration_ratio = frames_with_8_plus / total_frames if total_frames > 0 else 0.0
    
    # Calculate second utility: duration ratio for exactly 8 pills
    duration_ratio_exactly_8 = frames_with_exactly_8 / total_frames if total_frames > 0 else 0.0
    
    # Calculate statistics
    count_array = np.array(count_history)
    stats = {
        "total_frames": total_frames,
        "ever_reached_8_plus": ever_reached_8_plus,
        "binary_score": binary_score,
        "frames_with_8_plus": frames_with_8_plus,  # Keep for diagnostic
        "duration_ratio": duration_ratio,  # Keep for diagnostic
        "accuracy_percent": duration_ratio * 100.0,  # Keep for diagnostic
        "frames_with_exactly_8": frames_with_exactly_8,  # Second utility: count of frames with exactly 8 pills
        "duration_ratio_exactly_8": duration_ratio_exactly_8,  # Second utility: ratio of frames with exactly 8 pills
        "accuracy_percent_exactly_8": duration_ratio_exactly_8 * 100.0,  # Second utility: percentage
        "min_count": int(count_array.min()),
        "max_count": int(count_array.max()),
        "mean_count": float(count_array.mean()),
        "median_count": float(np.median(count_array)),
        "std_count": float(count_array.std()),
        "color_filter_enabled": use_color
    }
    
    return stats

def main():
    ap = argparse.ArgumentParser(description="Evaluate a single video with temporal tracking and HSV color matching")
    ap.add_argument("video", help="Path to video file")
    ap.add_argument("--settings", default="", help="JSON settings file")
    ap.add_argument("--temporal-params", default="", help="JSON file with temporal parameters (overrides settings)")
    ap.add_argument("--no-vis", action="store_true", help="Disable visualization (faster)")
    ap.add_argument("--max-frames", type=int, default=None, help="Override max_frames")
    ap.add_argument("--cluster-distance", type=float, default=None, help="Override cluster_distance")
    ap.add_argument("--min-ratio", type=float, default=None, help="Override min_detection_ratio")
    
    # Color matching arguments
    ap.add_argument("--enable-color", action="store_true", help="Enable HSV color filtering")
    ap.add_argument("--target-color-bgr", nargs=3, type=int, default=None, 
                    help="Target color in BGR format (e.g., 0 100 255 for orange)")
    ap.add_argument("--target-color-hsv", nargs=3, type=int, default=None,
                    help="Target color in HSV format (e.g., 20 200 200 for orange)")
    ap.add_argument("--hue-tolerance", type=int, default=None, help="Hue tolerance in degrees (default: 10)")
    ap.add_argument("--sat-range", nargs=2, type=int, default=None, help="Saturation range min max (default: 50 255)")
    ap.add_argument("--val-range", nargs=2, type=int, default=None, help="Value range min max (default: 50 255)")
    
    args = ap.parse_args()
    
    # Load settings
    cfg = load_settings(args.settings)
    
    # Load temporal parameters if provided
    if args.temporal_params and Path(args.temporal_params).exists():
        with open(args.temporal_params, "r") as f:
            temporal_params = json.load(f)
        cfg["temporal"].update(temporal_params)
        print(f"Loaded temporal parameters from: {args.temporal_params}")
    
    # Override temporal parameters from command line
    if args.max_frames is not None:
        cfg["temporal"]["max_frames"] = args.max_frames
    if args.cluster_distance is not None:
        cfg["temporal"]["cluster_distance"] = args.cluster_distance
    if args.min_ratio is not None:
        cfg["temporal"]["min_detection_ratio"] = args.min_ratio
    
    # Handle color matching arguments
    if args.enable_color:
        cfg["blob"]["use_color_filter"] = True
    
    if args.target_color_bgr:
        # Convert BGR to HSV
        bgr_color = np.uint8([[list(args.target_color_bgr)]])
        hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
        cfg["blob"]["target_color_hsv"] = [int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])]
        print(f"Converted BGR {args.target_color_bgr} to HSV {cfg['blob']['target_color_hsv']}")
    
    if args.target_color_hsv:
        cfg["blob"]["target_color_hsv"] = list(args.target_color_hsv)
    
    if args.hue_tolerance is not None:
        cfg["blob"]["hue_tolerance"] = args.hue_tolerance
    
    if args.sat_range:
        cfg["blob"]["sat_range"] = list(args.sat_range)
    
    if args.val_range:
        cfg["blob"]["val_range"] = list(args.val_range)
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        return
    
    # Evaluate video
    stats = evaluate_video_with_visualization(video_path, cfg, show_vis=not args.no_vis)
    
    if stats is None:
        return
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Video: {video_path.name}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"\nBinary Score: {stats['binary_score']:.0f} ({'PASS - Reached >=8' if stats['binary_score'] == 1.0 else 'FAIL - Never reached >=8'})")
    print(f"\nDiagnostic Info:")
    print(f"  Frames with >= 8 pills: {stats['frames_with_8_plus']}")
    print(f"  Duration Ratio: {stats['duration_ratio']:.2%}")
    print(f"  Accuracy: {stats['accuracy_percent']:.2f}%")
    print(f"\nSecond Utility (Exactly 8 Pills):")
    print(f"  Frames with exactly 8 pills: {stats['frames_with_exactly_8']}")
    print(f"  Duration Ratio: {stats['duration_ratio_exactly_8']:.2%}")
    print(f"  Accuracy: {stats['accuracy_percent_exactly_8']:.2f}%")
    print(f"\nCount Statistics:")
    print(f"  Min: {stats['min_count']}")
    print(f"  Max: {stats['max_count']}")
    print(f"  Mean: {stats['mean_count']:.2f}")
    print(f"  Median: {stats['median_count']:.2f}")
    print(f"  Std Dev: {stats['std_count']:.2f}")
    if stats['color_filter_enabled']:
        print(f"\nColor Filter: ENABLED")
        print(f"  Target HSV: {cfg['blob']['target_color_hsv']}")
        print(f"  Hue Tolerance: {cfg['blob']['hue_tolerance']}°")
        print(f"  Saturation Range: {cfg['blob']['sat_range']}")
        print(f"  Value Range: {cfg['blob']['val_range']}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

