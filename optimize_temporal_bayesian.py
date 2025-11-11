# optimize_temporal_bayesian.py
"""
Parameter optimizer using Bayesian optimization to find optimal temporal tracking and color matching parameters.
Tests combinations of max_frames, cluster_distance, min_detection_ratio, and color parameters on OK/Defect videos.

Requires: pip install scikit-optimize
"""
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from collections import deque

try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real
    from skopt.utils import use_named_args
except ImportError:
    print("ERROR: scikit-optimize is required. Install with: pip install scikit-optimize")
    exit(1)

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
        color_mask: Color mask used (for visualization), or None if not using color filter
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
        # Color matching parameters (optional)
        "use_color_filter": False,
        "target_color_hsv": [20, 200, 200],  # Default: orange
        "hue_tolerance": 10,
        "sat_range": [50, 255],
        "val_range": [50, 255]
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

# ---------- Evaluation Functions ----------
def evaluate_video(video_path, cfg, max_frames, cluster_distance, min_detection_ratio, 
                   hue_tolerance=None, sat_range_min=None, val_range_min=None, show_vis=False, is_ok=True):
    """
    Evaluate a single video with given temporal parameters.
    
    For OK videos:
    - Exactly 8 pills = best (score 1.0)
    - More than 8 pills = OK but not preferred (score 0.85)
    - Less than 8 pills = bad (score 0.0)
    
    For Defect videos:
    - 8 or more pills = very bad (score 0.0)
    - Less than 8 pills = good (score 1.0)
    
    Returns weighted score (0-1, higher is better).
    
    Args:
        show_vis: If True, display video processing in real-time
        is_ok: True for OK videos, False for Defect videos
    """
    warp_w = cfg["warp"]["width"]
    warp_h = cfg["warp"]["height"]
    M = None
    if cfg["warp"]["quad"]:
        M = load_quad(cfg["warp"]["quad"], warp_w, warp_h)
    
    detector = build_blob_params(cfg["blob"])
    tracker = TemporalPillTracker(
        max_frames=int(max_frames),
        cluster_distance=float(cluster_distance),
        min_detection_ratio=float(min_detection_ratio)
    )
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"      WARNING: Cannot open {video_path}")
        return 0.0
    
    # Get total frame count for progress
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    total_frames = 0
    frames_with_8_plus = 0
    last_progress_print = 0
    
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
        
        # Apply color filtering if parameters provided
        if hue_tolerance is not None and sat_range_min is not None and val_range_min is not None:
            cfg["blob"]["use_color_filter"] = True
            cfg["blob"]["hue_tolerance"] = int(hue_tolerance)
            cfg["blob"]["sat_range"] = [int(sat_range_min), 255]
            cfg["blob"]["val_range"] = [int(val_range_min), 255]
            # Ensure target_color_hsv is set (should be set in optimize_parameters, but double-check)
            if "target_color_hsv" not in cfg["blob"]:
                cfg["blob"]["target_color_hsv"] = [20, 200, 200]  # Fallback default
        
        # Detect (with or without color masking)
        color_mask = None
        if cfg["blob"].get("use_color_filter", False):
            # Debug: verify color filter is enabled
            if show_vis and total_frames == 0:
                print(f"      Color filter enabled: target_hsv={cfg['blob'].get('target_color_hsv')}, "
                      f"hue_tol={cfg['blob'].get('hue_tolerance')}, "
                      f"sat_range={cfg['blob'].get('sat_range')}, "
                      f"val_range={cfg['blob'].get('val_range')}")
            kps, color_mask = detect_with_color_mask(warped, gray_w, detector, cfg)
        else:
            kps = detector.detect(gray_w)
        tracker.add_frame(kps)
        stable_count = tracker.get_stable_count()
        current_count = len(kps)
        
        total_frames += 1
        
        # Calculate frame score based on count
        if is_ok:
            # OK video: exactly 8 is best, >8 is OK but not preferred, <8 is bad
            if stable_count == 8:
                frame_score = 1.0  # Perfect: exactly 8 pills
            elif stable_count > 8:
                frame_score = 0.85  # OK but not preferred: more than 8 pills
            else:
                frame_score = 0.0  # Bad: less than 8 pills
        else:
            # Defect video: want < 8 pills
            if stable_count >= 8:
                frame_score = 0.0  # Very bad: detected 8+ pills in defect video
            else:
                frame_score = 1.0  # Good: less than 8 pills
        
        frames_with_8_plus += frame_score  # Accumulate scores instead of binary count
        
        # Print progress every 10% of video
        if total_video_frames > 0:
            progress_pct = int((total_frames / total_video_frames) * 100)
            if progress_pct >= last_progress_print + 10:
                last_progress_print = progress_pct
                if not show_vis:  # Only print if not showing visualization
                    print(f"        Progress: {progress_pct}% ({total_frames}/{total_video_frames} frames)")
        
        # Real-time visualization
        if show_vis:
            vis = warped.copy()
            
            # Draw color mask overlay (semi-transparent) if available
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
            
            # Show video name and progress
            video_name = Path(video_path).name
            cv2.putText(vis, video_name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show current score
            if total_frames > 0:
                current_score = frames_with_8_plus / total_frames
                score_text = f"Score: {current_score:.3f}"
                cv2.putText(vis, score_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Optimization - Video Processing", vis)
            # Small delay to allow visualization, but don't block
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                # Don't break, just disable visualization for remaining frames
                show_vis = False
    
    cap.release()
    if show_vis:
        cv2.destroyAllWindows()
    
    if total_frames == 0:
        return 0.0
    
    # Return average score per frame
    return frames_with_8_plus / total_frames

def evaluate_parameter_set(max_frames, cluster_distance, min_detection_ratio, cfg, ok_videos, defect_videos, 
                           hue_tolerance=None, sat_range_min=None, val_range_min=None, 
                           show_vis=False, iteration_num=None):
    """
    Evaluate a parameter set on all videos.
    Returns score (0-1, higher is better):
    - OK videos: prefer exactly 8 pills (penalty for >8 or <8)
    - Defect videos: prefer <8 pills (penalty for >=8)
    
    Args:
        show_vis: If True, show real-time video processing (only for first video of each type)
        iteration_num: Current iteration number for progress display
    """
    # Evaluate OK videos (prefer exactly 8 pills)
    ok_scores = []
    total_ok = len(ok_videos)
    for i, video_path in enumerate(ok_videos):
        # Only show visualization for first OK video if enabled
        vis_enabled = show_vis and i == 0
        if iteration_num is not None:
            print(f"    Processing OK video {i+1}/{total_ok}: {Path(video_path).name}")
        ratio = evaluate_video(video_path, cfg, max_frames, cluster_distance, min_detection_ratio,
                              hue_tolerance=hue_tolerance, sat_range_min=sat_range_min, val_range_min=val_range_min,
                              show_vis=vis_enabled, is_ok=True)
        ok_scores.append(ratio)
        if iteration_num is not None:
            print(f"      → Score: {ratio:.3f} ({ratio:.1%})")
    
    # Evaluate Defect videos (want < 8 pills, score 1.0; >= 8 pills, score 0.0)
    defect_scores = []
    total_defect = len(defect_videos)
    for i, video_path in enumerate(defect_videos):
        # Only show visualization for first Defect video if enabled
        vis_enabled = show_vis and i == 0
        if iteration_num is not None:
            print(f"    Processing Defect video {i+1}/{total_defect}: {Path(video_path).name}")
        ratio = evaluate_video(video_path, cfg, max_frames, cluster_distance, min_detection_ratio,
                              hue_tolerance=hue_tolerance, sat_range_min=sat_range_min, val_range_min=val_range_min,
                              show_vis=vis_enabled, is_ok=False)
        defect_scores.append(ratio)
        if iteration_num is not None:
            print(f"      → Score: {ratio:.3f} ({ratio:.1%})")
    
    # Calculate composite score
    # Both OK and Defect scores are already normalized (0-1, higher is better)
    avg_ok = np.mean(ok_scores) if ok_scores else 0.0
    avg_defect = np.mean(defect_scores) if defect_scores else 0.0
    
    # Score: weighted combination
    # OK weight: 0.6, Defect weight: 0.4 (defect detection is critical)
    # Both scores are already "higher is better", so no need to invert defect
    score = 0.6 * avg_ok + 0.4 * avg_defect
    
    return score, avg_ok, avg_defect

# ---------- Bayesian Optimization ----------
def optimize_parameters(cfg, ok_videos, defect_videos, n_calls=50, random_state=42, show_vis=False, optimize_color=False, target_color_hsv=None):
    """
    Use Bayesian optimization to find optimal temporal and color parameters.
    
    Args:
        show_vis: If True, show real-time video processing during optimization
        optimize_color: If True, optimize color matching parameters
        target_color_hsv: Target HSV color [H, S, V] (required if optimize_color=True)
    """
    print(f"\n{'='*60}")
    print(f"Optimizing Parameters")
    print(f"{'='*60}")
    print(f"  OK videos: {len(ok_videos)}")
    print(f"  Defect videos: {len(defect_videos)}")
    print(f"  Total videos per iteration: {len(ok_videos) + len(defect_videos)}")
    print(f"  Optimization iterations: {n_calls}")
    print(f"  Total video evaluations: {n_calls * (len(ok_videos) + len(defect_videos))}")
    print(f"  Color optimization: {'ENABLED' if optimize_color else 'DISABLED'}")
    if optimize_color and target_color_hsv:
        print(f"  Target color HSV: {target_color_hsv}")
    if show_vis:
        print(f"  Visualization: ENABLED (press 'q' in video window to disable)")
    print(f"{'='*60}\n")
    
    # Set target color if provided
    if optimize_color:
        if target_color_hsv is None:
            target_color_hsv = cfg["blob"].get("target_color_hsv", [20, 200, 200])
        cfg["blob"]["target_color_hsv"] = target_color_hsv
    
    # Define search space
    if optimize_color:
        # Optimize temporal + color parameters (6 parameters)
        space = [
            Integer(5, 45, name='max_frames'),           # 5-45 frames
            Real(15.0, 40.0, name='cluster_distance'),  # 15-40 pixels
            Real(0.05, 0.4, name='min_detection_ratio'), # 0.05-0.4 (5%-40%)
            Integer(5, 30, name='hue_tolerance'),        # 5-30 degrees
            Integer(0, 150, name='sat_range_min'),       # 0-150 (saturation min)
            Integer(0, 150, name='val_range_min')         # 0-150 (value/brightness min)
        ]
    else:
        # Optimize only temporal parameters (3 parameters)
        space = [
            Integer(5, 45, name='max_frames'),           # 5-45 frames
            Real(15.0, 40.0, name='cluster_distance'),  # 15-40 pixels
            Real(0.05, 0.4, name='min_detection_ratio') # 0.05-0.4 (5%-40%)
        ]
    
    # Track best results
    best_score = -1.0
    best_params = None
    best_ok_ratio = 0.0
    best_defect_ratio = 0.0
    
    iteration_counter = [0]  # Use list to allow modification in nested function
    
    if optimize_color:
        @use_named_args(dimensions=space)
        def objective(max_frames, cluster_distance, min_detection_ratio, hue_tolerance, sat_range_min, val_range_min):
            nonlocal best_score, best_params, best_ok_ratio, best_defect_ratio
            
            iteration_counter[0] += 1
            current_iter = iteration_counter[0]
            
            print(f"\n{'='*60}")
            print(f"Iteration {current_iter}/{n_calls}")
            print(f"  Testing: max_frames={int(max_frames)}, "
                  f"cluster_dist={cluster_distance:.1f}, "
                  f"min_ratio={min_detection_ratio:.3f}")
            print(f"  Color: hue_tol={int(hue_tolerance)}, "
                  f"sat_min={int(sat_range_min)}, "
                  f"val_min={int(val_range_min)}")
            print(f"{'='*60}")
            
            # Ensure target_color_hsv is set in cfg before evaluation
            if optimize_color and target_color_hsv:
                cfg["blob"]["target_color_hsv"] = target_color_hsv
            
            score, ok_ratio, defect_ratio = evaluate_parameter_set(
                max_frames, cluster_distance, min_detection_ratio,
                cfg, ok_videos, defect_videos,
                hue_tolerance=hue_tolerance, sat_range_min=sat_range_min, val_range_min=val_range_min,
                show_vis=show_vis, iteration_num=current_iter
            )
            
            # Track best
            is_best = score > best_score
            if is_best:
                best_score = score
                best_params = {
                    "max_frames": int(max_frames),
                    "cluster_distance": float(cluster_distance),
                    "min_detection_ratio": float(min_detection_ratio),
                    "hue_tolerance": int(hue_tolerance),
                    "sat_range_min": int(sat_range_min),
                    "val_range_min": int(val_range_min)
                }
                best_ok_ratio = ok_ratio
                best_defect_ratio = defect_ratio
            
            print(f"\n  Results:")
            print(f"    Score: {score:.4f} {'★ BEST' if is_best else ''}")
            print(f"    OK avg score: {ok_ratio:.3f} ({ok_ratio:.1%})")
            print(f"    Defect avg score: {defect_ratio:.3f} ({defect_ratio:.1%})")
            if is_best:
                print(f"    → New best parameters found!")
            
            # Minimize negative score (gp_minimize minimizes)
            return -score
    else:
        @use_named_args(dimensions=space)
        def objective(max_frames, cluster_distance, min_detection_ratio):
            nonlocal best_score, best_params, best_ok_ratio, best_defect_ratio
            
            iteration_counter[0] += 1
            current_iter = iteration_counter[0]
            
            print(f"\n{'='*60}")
            print(f"Iteration {current_iter}/{n_calls}")
            print(f"  Testing: max_frames={int(max_frames)}, "
                  f"cluster_dist={cluster_distance:.1f}, "
                  f"min_ratio={min_detection_ratio:.3f}")
            print(f"{'='*60}")
            
            score, ok_ratio, defect_ratio = evaluate_parameter_set(
                max_frames, cluster_distance, min_detection_ratio,
                cfg, ok_videos, defect_videos, show_vis=show_vis, iteration_num=current_iter
            )
            
            # Track best
            is_best = score > best_score
            if is_best:
                best_score = score
                best_params = {
                    "max_frames": int(max_frames),
                    "cluster_distance": float(cluster_distance),
                    "min_detection_ratio": float(min_detection_ratio)
                }
                best_ok_ratio = ok_ratio
                best_defect_ratio = defect_ratio
            
            print(f"\n  Results:")
            print(f"    Score: {score:.4f} {'★ BEST' if is_best else ''}")
            print(f"    OK avg score: {ok_ratio:.3f} ({ok_ratio:.1%})")
            print(f"    Defect avg score: {defect_ratio:.3f} ({defect_ratio:.1%})")
            if is_best:
                print(f"    → New best parameters found!")
            
            # Minimize negative score (gp_minimize minimizes)
            return -score
    
    # Run optimization
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        random_state=random_state,
        n_initial_points=10,
        acq_func='EI'  # Expected Improvement
    )
    
    # Extract best parameters
    if optimize_color:
        best_max_frames = int(result.x[0])
        best_cluster_distance = float(result.x[1])
        best_min_ratio = float(result.x[2])
        best_hue_tol = int(result.x[3])
        best_sat_min = int(result.x[4])
        best_val_min = int(result.x[5])
        
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(f"Best Parameters:")
        print(f"  Temporal:")
        print(f"    max_frames: {best_max_frames}")
        print(f"    cluster_distance: {best_cluster_distance:.2f}")
        print(f"    min_detection_ratio: {best_min_ratio:.3f}")
        print(f"  Color:")
        print(f"    hue_tolerance: {best_hue_tol}")
        print(f"    sat_range_min: {best_sat_min}")
        print(f"    val_range_min: {best_val_min}")
        print(f"    target_color_hsv: {target_color_hsv}")
        print(f"\nBest Score: {best_score:.4f}")
        print(f"  OK videos avg score: {best_ok_ratio:.3f} ({best_ok_ratio:.1%})")
        print(f"  Defect videos avg score: {best_defect_ratio:.3f} ({best_defect_ratio:.1%})")
        print(f"{'='*60}\n")
        
        return {
            "max_frames": best_max_frames,
            "cluster_distance": best_cluster_distance,
            "min_detection_ratio": best_min_ratio,
            "hue_tolerance": best_hue_tol,
            "sat_range_min": best_sat_min,
            "val_range_min": best_val_min,
            "target_color_hsv": target_color_hsv
        }
    else:
        best_max_frames = int(result.x[0])
        best_cluster_distance = float(result.x[1])
        best_min_ratio = float(result.x[2])
        
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(f"Best Parameters:")
        print(f"  max_frames: {best_max_frames}")
        print(f"  cluster_distance: {best_cluster_distance:.2f}")
        print(f"  min_detection_ratio: {best_min_ratio:.3f}")
        print(f"\nBest Score: {best_score:.4f}")
        print(f"  OK videos avg score: {best_ok_ratio:.3f} ({best_ok_ratio:.1%})")
        print(f"  Defect videos avg score: {best_defect_ratio:.3f} ({best_defect_ratio:.1%})")
        print(f"{'='*60}\n")
        
        return {
            "max_frames": best_max_frames,
            "cluster_distance": best_cluster_distance,
            "min_detection_ratio": best_min_ratio
        }

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Optimize temporal tracking and color matching parameters using Bayesian optimization")
    ap.add_argument("--settings", default="", help="JSON settings file (for warp/preprocess/blob)")
    ap.add_argument("--n-calls", type=int, default=150, help="Number of optimization iterations (default: 50)")
    ap.add_argument("--auto-run", action="store_true", help="Run optimization automatically without confirmation")
    ap.add_argument("--show-vis", action="store_true", help="Show real-time video processing during optimization")
    
    # Color optimization arguments
    ap.add_argument("--optimize-color", action="store_true", help="Enable color parameter optimization")
    ap.add_argument("--target-color-hsv", nargs=3, type=int, default=None,
                    help="Target color in HSV format [H S V] (required if --optimize-color, e.g., 20 200 200 for orange)")
    ap.add_argument("--target-color-bgr", nargs=3, type=int, default=None,
                    help="Target color in BGR format [B G R] (will be converted to HSV)")
    
    args = ap.parse_args()
    
    # Load settings
    cfg = load_settings(args.settings)
    
    # Find videos
    data_dir = Path("data")
    ok_videos = list(data_dir.glob("OK/*.mp4"))
    defect_videos = list(data_dir.glob("Defect/*.mp4"))
    
    if not ok_videos and not defect_videos:
        print("ERROR: No videos found in data/OK/ or data/Defect/")
        return
    
    if not ok_videos:
        print("WARNING: No OK videos found. Optimization may be less effective.")
    if not defect_videos:
        print("WARNING: No Defect videos found. Optimization may be less effective.")
    
    print(f"Found {len(ok_videos)} OK videos and {len(defect_videos)} Defect videos")
    
    # Handle color optimization setup
    target_color_hsv = None
    if args.optimize_color:
        if args.target_color_bgr:
            # Convert BGR to HSV
            bgr_color = np.uint8([[list(args.target_color_bgr)]])
            hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
            target_color_hsv = [int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])]
            print(f"Converted BGR {args.target_color_bgr} to HSV {target_color_hsv}")
        elif args.target_color_hsv:
            target_color_hsv = list(args.target_color_hsv)
        else:
            # Use default from config
            target_color_hsv = cfg["blob"].get("target_color_hsv", [20, 200, 200])
            print(f"Using default target color HSV: {target_color_hsv}")
    
    if not args.auto_run:
        opt_type = "temporal + color" if args.optimize_color else "temporal"
        response = input(f"\nRun {opt_type} optimization with {args.n_calls} iterations? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Run optimization
    best_params = optimize_parameters(
        cfg, ok_videos, defect_videos, 
        n_calls=args.n_calls, 
        show_vis=args.show_vis,
        optimize_color=args.optimize_color,
        target_color_hsv=target_color_hsv
    )
    
    # Save results
    if args.optimize_color:
        output_path = Path("optimized_params_color.json")
    else:
        output_path = Path("optimized_temporal_params.json")
    
    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Saved optimal parameters to: {output_path}")

if __name__ == "__main__":
    main()
