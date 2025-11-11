# eval_video_temporal.py
"""
Single video evaluator with temporal tracking.
Processes a video and outputs duration-based accuracy percentage.
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
        "min_thr": 10, "max_thr": 220, "thr_step": 10
    },
    "temporal": {
        "enabled": True,
        "max_frames": 10,
        "cluster_distance": 30.0,
        "min_detection_ratio": 0.3
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
    Evaluate a single video with temporal tracking and optional visualization.
    Returns duration ratio (frames with >= 8 pills / total frames) and statistics.
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
    frames_with_8_plus = 0
    count_history = []
    
    print(f"Processing video: {video_path.name}")
    print(f"Temporal params: max_frames={max_frames}, cluster_dist={cluster_distance:.1f}, min_ratio={min_detection_ratio:.2f}")
    
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
        
        # Detect
        kps = detector.detect(gray_w)
        tracker.add_frame(kps)
        stable_count = tracker.get_stable_count()
        current_count = len(kps)
        
        total_frames += 1
        count_history.append(stable_count)
        
        if stable_count >= 8:
            frames_with_8_plus += 1
        
        # Visualization
        if show_vis:
            vis = cv2.drawKeypoints(warped, kps, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # Color coding based on stable count
            if stable_count >= 8:
                count_color = (0, 255, 0)  # Green - OK
            elif stable_count >= 6:
                count_color = (0, 165, 255)  # Orange - partial
            else:
                count_color = (0, 0, 255)  # Red - defect
            
            count_text = f"stable={stable_count} (current={current_count})"
            cv2.putText(vis, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, count_color, 2)
            
            # Show duration ratio
            if total_frames > 0:
                duration_ratio = frames_with_8_plus / total_frames
                ratio_text = f"Duration ratio: {duration_ratio:.1%}"
                cv2.putText(vis, ratio_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Video Evaluation", vis)
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
    
    duration_ratio = frames_with_8_plus / total_frames
    
    # Calculate statistics
    count_array = np.array(count_history)
    stats = {
        "total_frames": total_frames,
        "frames_with_8_plus": frames_with_8_plus,
        "duration_ratio": duration_ratio,
        "accuracy_percent": duration_ratio * 100.0,
        "min_count": int(count_array.min()),
        "max_count": int(count_array.max()),
        "mean_count": float(count_array.mean()),
        "median_count": float(np.median(count_array)),
        "std_count": float(count_array.std())
    }
    
    return stats

def main():
    ap = argparse.ArgumentParser(description="Evaluate a single video with temporal tracking")
    ap.add_argument("video", help="Path to video file")
    ap.add_argument("--settings", default="", help="JSON settings file")
    ap.add_argument("--temporal-params", default="", help="JSON file with temporal parameters (overrides settings)")
    ap.add_argument("--no-vis", action="store_true", help="Disable visualization (faster)")
    ap.add_argument("--max-frames", type=int, default=None, help="Override max_frames")
    ap.add_argument("--cluster-distance", type=float, default=None, help="Override cluster_distance")
    ap.add_argument("--min-ratio", type=float, default=None, help="Override min_detection_ratio")
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
    print(f"Frames with >= 8 pills: {stats['frames_with_8_plus']}")
    print(f"\nDuration Ratio: {stats['duration_ratio']:.2%}")
    print(f"Accuracy: {stats['accuracy_percent']:.2f}%")
    print(f"\nCount Statistics:")
    print(f"  Min: {stats['min_count']}")
    print(f"  Max: {stats['max_count']}")
    print(f"  Mean: {stats['mean_count']:.2f}")
    print(f"  Median: {stats['median_count']:.2f}")
    print(f"  Std Dev: {stats['std_count']:.2f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

