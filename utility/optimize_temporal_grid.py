import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from itertools import product

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

def create_color_mask(warped_bgr, target_color_hsv, hue_tolerance=10, sat_range=(50, 255), val_range=(50, 255)):
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

DEFECT_EXPECTED_COUNTS = {
    "clip_1762491325.mp4": 7,
    "clip_1762491458.mp4": 7,
    "clip_1762491749.mp4": 6,
    "clip_1762491905.mp4": 6,
    "clip_1762492383.mp4": 5,
    "clip_1762492645.mp4": 4,
    "clip_1762492660.mp4": 4,
    "clip_1762492699.mp4": 4,
    "clip_1762492714.mp4": 4,
    "clip_1762493073.mp4": 7,
    "clip_1762493098.mp4": 8,
    "clip_1762493114.mp4": 8,
    "clip_1762493127.mp4": 6,
    "clip_1762493150.mp4": 7,
    "clip_1762493171.mp4": 7,
    "clip_1762493184.mp4": 7,
    "clip_1762493198.mp4": 7,
    "clip_1762493905.mp4": 6,
    "clip_1762493929.mp4": 6,
    "clip_1762493946.mp4": 6,
    "clip_1762493961.mp4": 6,
}

OK_EXPECTED_COUNT = 8  # All OK videos should detect 8 pills

def evaluate_video(video_path, cfg, max_frames, cluster_distance, min_detection_ratio, 
                   hue_tolerance=None, sat_range_min=None, val_range_min=None, show_vis=False, is_ok=True,
                   expected_count=None):
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
        return 0.0, 0
    
    # Get total frame count for progress
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    total_frames = 0
    last_progress_print = 0
    
    # Determine expected count
    if expected_count is None:
        if is_ok:
            expected_count = OK_EXPECTED_COUNT
        else:
            video_name = Path(video_path).name
            expected_count = DEFECT_EXPECTED_COUNTS.get(video_name, 8)  # Default to 8 if not found
    
    # Track maximum detected count
    max_detected_count = 0
    
    # Track frames where count matches expected (secondary heuristic for tiebreaking)
    frames_with_exactly_expected = 0
    
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
            kps, color_mask = detect_with_color_mask(warped, gray_w, detector, cfg)
        else:
            kps = detector.detect(gray_w)
        tracker.add_frame(kps)
        stable_count = tracker.get_stable_count()
        current_count = len(kps)
        
        total_frames += 1
        
        # Track maximum detected count
        if stable_count > max_detected_count:
            max_detected_count = stable_count
        
        # Track frames with exactly expected count (secondary heuristic for tiebreaking)
        if stable_count == expected_count:
            frames_with_exactly_expected += 1
        
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
            
            # Show max detected and expected
            status_text = f"Max: {max_detected_count}, Expected: {expected_count}"
            status_color = (0, 255, 0) if max_detected_count == expected_count else (0, 0, 255)
            cv2.putText(vis, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            cv2.imshow("Grid Search - Video Processing", vis)
            # Small delay to allow visualization, but don't block
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                # Don't break, just disable visualization for remaining frames
                show_vis = False
    
    cap.release()
    if show_vis:
        cv2.destroyAllWindows()
    
    if total_frames == 0:
        return 0.0, 0  # Return (score, frames_with_exactly_expected)
    
    # Score based on how close max detected is to expected count
    if max_detected_count == expected_count:
        score = 1.0  # Perfect match
    else:
        # Penalty for deviation: linear penalty based on difference
        difference = abs(max_detected_count - expected_count)
        # Score decreases linearly: difference of 1 = 0.5, difference of 2 = 0.0, etc.
        score = max(0.0, 1.0 - (difference * 0.5))
    
    # Return both score and count of frames with exactly expected count
    return score, frames_with_exactly_expected

def evaluate_parameter_set(max_frames, cluster_distance, min_detection_ratio, cfg, ok_videos, defect_videos, 
                           hue_tolerance=None, sat_range_min=None, val_range_min=None, 
                           show_vis=False, iteration_num=None, total_iterations=None):
    # Evaluate OK videos (should detect 8)
    ok_scores = []
    ok_frames_exactly_expected = []  # Track frames with exactly expected count for OK videos
    total_ok = len(ok_videos)
    for i, video_path in enumerate(ok_videos):
        # Only show visualization for first OK video if enabled
        vis_enabled = show_vis and i == 0
        if iteration_num is not None:
            print(f"    Processing OK video {i+1}/{total_ok}: {Path(video_path).name}")
        score, frames_exactly_expected = evaluate_video(video_path, cfg, max_frames, cluster_distance, min_detection_ratio,
                              hue_tolerance=hue_tolerance, sat_range_min=sat_range_min, val_range_min=val_range_min,
                              show_vis=vis_enabled, is_ok=True, expected_count=OK_EXPECTED_COUNT)
        ok_scores.append(score)
        ok_frames_exactly_expected.append(frames_exactly_expected)
        if iteration_num is not None:
            status = "PASS" if score == 1.0 else f"Score: {score:.2f}"
            print(f"      → {status}, frames=={OK_EXPECTED_COUNT}: {frames_exactly_expected}")
    
    # Evaluate Defect videos (should match expected count for each video)
    defect_scores = []
    defect_frames_exactly_expected = []  # Track frames with exactly expected count for Defect videos
    total_defect = len(defect_videos)
    for i, video_path in enumerate(defect_videos):
        # Only show visualization for first Defect video if enabled
        vis_enabled = show_vis and i == 0
        video_name = Path(video_path).name
        expected_count = DEFECT_EXPECTED_COUNTS.get(video_name, 8)  # Default to 8 if not found
        if iteration_num is not None:
            print(f"    Processing Defect video {i+1}/{total_defect}: {video_name} (expected: {expected_count})")
        score, frames_exactly_expected = evaluate_video(video_path, cfg, max_frames, cluster_distance, min_detection_ratio,
                              hue_tolerance=hue_tolerance, sat_range_min=sat_range_min, val_range_min=val_range_min,
                              show_vis=vis_enabled, is_ok=False, expected_count=expected_count)
        defect_scores.append(score)
        defect_frames_exactly_expected.append(frames_exactly_expected)
        if iteration_num is not None:
            status = "PASS" if score == 1.0 else f"Score: {score:.2f}"
            print(f"      → {status}, frames=={expected_count}: {frames_exactly_expected}")
    
    # Calculate composite score
    avg_ok = np.mean(ok_scores) if ok_scores else 0.0
    avg_defect = np.mean(defect_scores) if defect_scores else 0.0
    
    score = 0.5 * avg_ok + 0.5 * avg_defect
    
    # Calculate total frames with exactly expected count (for tiebreaking)
    total_ok_frames_exactly_expected = sum(ok_frames_exactly_expected) if ok_frames_exactly_expected else 0
    total_defect_frames_exactly_expected = sum(defect_frames_exactly_expected) if defect_frames_exactly_expected else 0
    
    return score, avg_ok, avg_defect, total_ok_frames_exactly_expected, total_defect_frames_exactly_expected

def generate_grid_points(step_sizes=None):
    if step_sizes is None:
        step_sizes = {}
    
    # Default step sizes (optimized around best results: max_frames=5, cluster_distance=40, min_ratio=0.335)
    default_steps = {
        'max_frames': 1,               # Fine-grained steps around 5
        'cluster_distance': 2.0,      # Fine-grained steps around 40
        'min_detection_ratio': 0.005, # Fine-grained steps around 0.335
        'hue_tolerance': 20,           # Not used (color optimization disabled)
        'sat_range_min': 55,           # Not used (color optimization disabled)
        'val_range_min': 8             # Not used (color optimization disabled)
    }
    
    # Default ranges (focused around best optimization results, optimized for ~300 combinations)
    default_ranges = {
        'max_frames': (3, 7),              # Search around 5: 3-7 (5 values with step=1)
        'cluster_distance': (32.0, 44.0),  # Search around 40: 32-44 (7 values with step=2.0)
        'min_detection_ratio': (0.30, 0.35) # Search around 0.335: 0.30-0.35 (11 values with step=0.005)
    }
    # Total: 5 × 7 × 11 = 385 combinations (slightly over 300, but manageable)
    
    # Merge with user-provided step sizes and ranges
    steps = {**default_steps, **step_sizes}
    ranges = default_ranges.copy()
    if 'max_frames_range' in step_sizes:
        ranges['max_frames'] = step_sizes['max_frames_range']
    if 'cluster_distance_range' in step_sizes:
        ranges['cluster_distance'] = step_sizes['cluster_distance_range']
    if 'min_detection_ratio_range' in step_sizes:
        ranges['min_detection_ratio'] = step_sizes['min_detection_ratio_range']
    
    # Generate grid for 3 temporal parameters only (fine-grained around optimal values)
    max_frames_range = range(ranges['max_frames'][0], ranges['max_frames'][1] + 1, steps['max_frames'])
    cluster_dist_range = np.arange(ranges['cluster_distance'][0], ranges['cluster_distance'][1] + 0.1, steps['cluster_distance'])
    min_ratio_range = np.arange(ranges['min_detection_ratio'][0], ranges['min_detection_ratio'][1] + 0.01, steps['min_detection_ratio'])
    
    # Generate all combinations
    grid_points = list(product(
        max_frames_range,
        cluster_dist_range,
        min_ratio_range
    ))
    
    return grid_points

def optimize_parameters_grid(cfg, ok_videos, defect_videos, step_sizes=None, show_vis=False,
                             hue_tolerance=None, sat_range_min=None, val_range_min=None, target_color_hsv=None):
    print(f"\n{'='*60}")
    print(f"Grid Search Optimization (Temporal Parameters Only)")
    print(f"{'='*60}")
    print(f"  OK videos: {len(ok_videos)}")
    print(f"  Defect videos: {len(defect_videos)}")
    print(f"  Total videos per iteration: {len(ok_videos) + len(defect_videos)}")
    print(f"  Parameters: max_frames, cluster_distance, min_detection_ratio")
    if hue_tolerance is not None and sat_range_min is not None and val_range_min is not None:
        print(f"  Color mask: ENABLED (hue_tol={hue_tolerance}, sat_min={sat_range_min}, val_min={val_range_min})")
        if target_color_hsv:
            print(f"  Target color HSV: {target_color_hsv}")
    else:
        print(f"  Color mask: DISABLED")
    if show_vis:
        print(f"  Visualization: ENABLED (press 'q' in video window to disable)")
    
    # Set target color in config if provided
    if target_color_hsv:
        cfg["blob"]["target_color_hsv"] = target_color_hsv
    
    # Generate grid points (3 parameters only)
    grid_points = generate_grid_points(step_sizes=step_sizes)
    total_iterations = len(grid_points)
    print(f"  Total grid points: {total_iterations}")
    print(f"{'='*60}\n")
    
    # Track best results
    best_score = -1.0
    best_params = None
    best_ok_ratio = 0.0
    best_defect_ratio = 0.0
    best_ok_frames_exactly_expected = 0
    best_defect_frames_exactly_expected = 0
    
    # Evaluate all grid points
    for iteration, params in enumerate(grid_points, 1):
        max_frames, cluster_distance, min_detection_ratio = params
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{total_iterations}")
        print(f"  Testing: max_frames={int(max_frames)}, "
              f"cluster_dist={cluster_distance:.1f}, "
              f"min_ratio={min_detection_ratio:.3f}")
        print(f"{'='*60}")
        
        score, ok_ratio, defect_ratio, ok_frames_exactly_expected, defect_frames_exactly_expected = evaluate_parameter_set(
            max_frames, cluster_distance, min_detection_ratio,
            cfg, ok_videos, defect_videos,
            hue_tolerance=hue_tolerance, sat_range_min=sat_range_min, val_range_min=val_range_min,
            show_vis=show_vis, iteration_num=iteration, total_iterations=total_iterations
        )
        
        is_best = False
        if score > best_score:
            is_best = True
        elif score == best_score and best_score == 1.0:
            current_tiebreaker = ok_frames_exactly_expected + defect_frames_exactly_expected
            best_tiebreaker = best_ok_frames_exactly_expected + best_defect_frames_exactly_expected
            if current_tiebreaker > best_tiebreaker:
                is_best = True
        
        if is_best:
            best_score = score
            best_params = {
                "max_frames": int(max_frames),
                "cluster_distance": float(cluster_distance),
                "min_detection_ratio": float(min_detection_ratio)
            }
            best_ok_ratio = ok_ratio
            best_defect_ratio = defect_ratio
            best_ok_frames_exactly_expected = ok_frames_exactly_expected
            best_defect_frames_exactly_expected = defect_frames_exactly_expected
        
        print(f"\n  Results:")
        print(f"    Score: {score:.4f} {'★ BEST' if is_best else ''}")
        print(f"    OK avg score: {ok_ratio:.3f} ({ok_ratio:.1%} videos passed)")
        print(f"    Defect avg score: {defect_ratio:.3f} ({defect_ratio:.1%} videos passed)")
        print(f"    OK frames==expected: {ok_frames_exactly_expected} (tiebreaker)")
        print(f"    Defect frames==expected: {defect_frames_exactly_expected} (tiebreaker)")
        if is_best:
            print(f"    → New best parameters found!")
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"Grid Search Complete!")
    print(f"{'='*60}")
    print(f"Best Parameters:")
    print(f"  max_frames: {best_params['max_frames']}")
    print(f"  cluster_distance: {best_params['cluster_distance']:.2f}")
    print(f"  min_detection_ratio: {best_params['min_detection_ratio']:.3f}")
    print(f"\nBest Score: {best_score:.4f}")
    print(f"  OK videos avg score: {best_ok_ratio:.3f} ({best_ok_ratio:.1%})")
    print(f"  Defect videos avg score: {best_defect_ratio:.3f} ({best_defect_ratio:.1%})")
    print(f"\nTiebreaker metrics (frames matching expected counts):")
    print(f"  OK videos total frames=={OK_EXPECTED_COUNT}: {best_ok_frames_exactly_expected}")
    print(f"  Defect videos total frames==expected: {best_defect_frames_exactly_expected}")
    print(f"{'='*60}\n")
    
    return {
        "max_frames": best_params['max_frames'],
        "cluster_distance": best_params['cluster_distance'],
        "min_detection_ratio": best_params['min_detection_ratio']
    }

def main():
    ap = argparse.ArgumentParser(description="Optimize temporal tracking parameters using grid search")
    ap.add_argument("--settings", default="", help="JSON settings file (for warp/preprocess/blob)")
    ap.add_argument("--auto-run", action="store_true", help="Run optimization automatically without confirmation")
    ap.add_argument("--show-vis", action="store_true", help="Show real-time video processing during optimization")
    
    # Grid search step sizes (defaults optimized around best results: max_frames=5, cluster_distance=40, min_ratio=0.335)
    ap.add_argument("--step-max-frames", type=int, default=1, help="Step size for max_frames (default: 1, fine-grained around 5)")
    ap.add_argument("--step-cluster-distance", type=float, default=2.0, help="Step size for cluster_distance (default: 2.0, fine-grained around 40)")
    ap.add_argument("--step-min-ratio", type=float, default=0.005, help="Step size for min_detection_ratio (default: 0.005, fine-grained around 0.335)")
    
    # Search ranges (focused around best optimization results, optimized for ~300 combinations)
    ap.add_argument("--max-frames-range", nargs=2, type=int, default=[3, 7], 
                    help="Range for max_frames [min max] (default: 3 7, around optimal 5)")
    ap.add_argument("--cluster-distance-range", nargs=2, type=float, default=[32.0, 44.0],
                    help="Range for cluster_distance [min max] (default: 32.0 44.0, around optimal 40)")
    ap.add_argument("--min-ratio-range", nargs=2, type=float, default=[0.30, 0.35],
                    help="Range for min_detection_ratio [min max] (default: 0.30 0.35, around optimal 0.335)")
    
    # Fixed color mask parameters (used for all evaluations)
    ap.add_argument("--hue-tolerance", type=int, default=18,
                    help="Fixed hue tolerance for color mask (default: 18)")
    ap.add_argument("--sat-range-min", type=int, default=90,
                    help="Fixed saturation range minimum for color mask (default: 90)")
    ap.add_argument("--val-range-min", type=int, default=35,
                    help="Fixed value range minimum for color mask (default: 35)")
    ap.add_argument("--target-color-hsv", nargs=3, type=int, default=None,
                    help="Target color in HSV format [H S V] (required if using color mask, e.g., 19 124 101)")
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
    
    # Prepare step sizes and ranges
    step_sizes = {
        'max_frames': args.step_max_frames,
        'cluster_distance': args.step_cluster_distance,
        'min_detection_ratio': args.step_min_ratio,
        'max_frames_range': tuple(args.max_frames_range),
        'cluster_distance_range': tuple(args.cluster_distance_range),
        'min_detection_ratio_range': tuple(args.min_ratio_range)
    }
    
    # Calculate total iterations
    grid_points = generate_grid_points(step_sizes=step_sizes)
    total_iterations = len(grid_points)
    
    if not args.auto_run:
        print(f"\nGrid search will test {total_iterations} parameter combinations")
        print(f"Estimated time: ~{total_iterations * (len(ok_videos) + len(defect_videos)) * 0.1:.1f} minutes (rough estimate)")
        response = input(f"\nRun temporal parameter grid search? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Handle color optimization setup
    target_color_hsv = None
    if args.target_color_bgr:
        # Convert BGR to HSV
        bgr_color = np.uint8([[list(args.target_color_bgr)]])
        hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
        target_color_hsv = [int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])]
        print(f"Converted BGR {args.target_color_bgr} to HSV {target_color_hsv}")
    elif args.target_color_hsv:
        target_color_hsv = list(args.target_color_hsv)
    
    # Determine if color mask should be used
    use_color_mask = (args.hue_tolerance is not None and 
                     args.sat_range_min is not None and 
                     args.val_range_min is not None and
                     target_color_hsv is not None)
    
    if use_color_mask:
        print(f"\nColor mask enabled with fixed parameters:")
        print(f"  hue_tolerance: {args.hue_tolerance}")
        print(f"  sat_range_min: {args.sat_range_min}")
        print(f"  val_range_min: {args.val_range_min}")
        print(f"  target_color_hsv: {target_color_hsv}")
    else:
        print(f"\nColor mask disabled (no target color specified)")
        args.hue_tolerance = None
        args.sat_range_min = None
        args.val_range_min = None
    
    # Run optimization
    best_params = optimize_parameters_grid(
        cfg, ok_videos, defect_videos, 
        show_vis=args.show_vis,
        step_sizes=step_sizes,
        hue_tolerance=args.hue_tolerance,
        sat_range_min=args.sat_range_min,
        val_range_min=args.val_range_min,
        target_color_hsv=target_color_hsv
    )
    
    # Save results
    output_path = Path("optimized_temporal_params_grid.json")
    
    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Saved optimal parameters to: {output_path}")

if __name__ == "__main__":
    main()

