# count_pills_warp.py
import cv2, json, argparse, time, numpy as np
from pathlib import Path
from collections import deque

# ---------- Warp helpers ----------
def load_quad(json_path, warp_w, warp_h):
    with open(json_path, "r") as f:
        pts = np.array(json.load(f)["points"], dtype=np.float32)  # TL,TR,BR,BL
    dst = np.array([[0,0],[warp_w-1,0],[warp_w-1,warp_h-1],[0,warp_h-1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(pts, dst)

def warp_topdown(bgr, M, size):
    return cv2.warpPerspective(bgr, M, size)

# ---------- Blob detector ----------
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

# ---------- Preprocess (identical for camera & video) ----------
def apply_gamma(img, gamma):
    if abs(gamma - 1.0) < 1e-3: return img
    # LUT for speed
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
    # 1) digital zoom
    img = center_zoom(frame_bgr, pre["zoom"])
    # 2) optional rotate (keep tray aligned)
    if abs(pre["rotate_deg"]) > 1e-3:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), pre["rotate_deg"], 1.0)
        img = cv2.warpAffine(img, M, (w, h))
    # 3) brightness/contrast (alpha, beta)
    img = cv2.convertScaleAbs(img, alpha=pre["alpha"], beta=pre["beta"])
    # 4) gamma
    img = apply_gamma(img, pre["gamma"])
    # 5) simple white-balance gains
    img = apply_wb_gains(img, pre["wb"]) if pre["wb"] else img
    # Return both color + gray variants
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 6) CLAHE
    clahe = cv2.createCLAHE(clipLimit=pre["clahe_clip"], tileGridSize=(pre["clahe_tile"], pre["clahe_tile"]))
    gray = clahe.apply(gray)
    # 7) blur
    if pre["blur"] > 0:
        k = pre["blur"] + (1 - pre["blur"] % 2)  # odd
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return img, gray

# ---------- Temporal Aggregation ----------
class TemporalPillTracker:
    """
    Tracks pill detections across multiple frames to stabilize counts.
    Handles flickering by aggregating detections over time using spatial clustering.
    """
    def __init__(self, max_frames=30, cluster_distance=30.0, min_detection_ratio=0.3):
        """
        Args:
            max_frames: Number of recent frames to track (default: 30, min: 2)
            cluster_distance: Max pixel distance for clustering detections (default: 30.0)
            min_detection_ratio: Min ratio of frames a pill must appear in to count (default: 0.3 = 30%)
        """
        # Ensure max_frames is at least 2 for temporal tracking to work
        self.max_frames = max(2, max_frames) if max_frames > 0 else 2
        self.cluster_distance = cluster_distance
        self.min_detection_ratio = min_detection_ratio
        self.detection_history = deque(maxlen=self.max_frames)
        self.frame_counter = 0
    
    def add_frame(self, keypoints):
        """Add detections from current frame to history."""
        points = np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints], dtype=np.float32)
        self.detection_history.append((self.frame_counter, points))
        self.frame_counter += 1
    
    def get_stable_count(self):
        """
        Calculate stable pill count by clustering detections across frames.
        Returns the number of pills that appear frequently enough.
        """
        if len(self.detection_history) < 2:
            # Not enough frames yet, return current count
            if self.detection_history:
                return len(self.detection_history[-1][1])
            return 0
        
        # Collect all detection points with their frame indices
        all_points = []
        for frame_idx, points in self.detection_history:
            for pt in points:
                all_points.append((pt[0], pt[1], frame_idx))
        
        if not all_points:
            return 0
        
        # Convert to numpy for efficient computation
        points_array = np.array([(p[0], p[1]) for p in all_points], dtype=np.float32)
        frame_indices = np.array([p[2] for p in all_points], dtype=np.int32)
        
        # Cluster nearby detections (same pill across frames)
        clusters = self._cluster_points(points_array, frame_indices)
        
        # Count clusters that appear frequently enough
        min_frames_required = max(1, int(len(self.detection_history) * self.min_detection_ratio))
        stable_count = 0
        
        for cluster_frames in clusters:
            if len(cluster_frames) >= min_frames_required:
                stable_count += 1
        
        return stable_count
    
    def _cluster_points(self, points_array, frame_indices):
        """
        Cluster spatially nearby points using distance-based grouping.
        Returns list of sets, each containing frame indices where that cluster appears.
        """
        n = len(points_array)
        if n == 0:
            return []
        
        visited = np.zeros(n, dtype=bool)
        clusters = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Start new cluster
            cluster_indices = [i]
            visited[i] = True
            
            # Find all nearby points (within cluster_distance)
            for j in range(i + 1, n):
                if visited[j]:
                    continue
                
                dist = np.linalg.norm(points_array[i] - points_array[j])
                if dist < self.cluster_distance:
                    cluster_indices.append(j)
                    visited[j] = True
            
            # Extract frame indices for this cluster
            cluster_frames = set(frame_indices[cluster_indices])
            clusters.append(cluster_frames)
        
        return clusters
    
    def reset(self):
        """Reset tracking state."""
        self.detection_history.clear()
        self.frame_counter = 0

# ---------- Settings ----------
DEFAULTS = {
    "warp": {"quad": "", "width": 600, "height": 300},
    "preprocess": {
        "zoom": 2.5, "rotate_deg": 0.0,
        "alpha": 1.0, "beta": 0.0,           # contrast, brightness
        "gamma": 1.0,
        "wb": [1.0, 1.0, 1.0],               # B,G,R gains; set null to disable
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
        # TUNING GUIDE:
        # - max_frames: More frames = more stable but slower response (20-45 recommended)
        # - cluster_distance: Larger = more lenient spatial matching (25-40px recommended)
        # - min_detection_ratio: Lower = count pills appearing less often (0.2-0.4 recommended)
        # To disable temporal tracking: set "enabled": False OR "max_frames": 0
        "max_frames": 10,              # Number of frames to track (recommended: 20-45)
        "cluster_distance": 30.0,       # Pixel distance for clustering (recommended: 25-40)
        "min_detection_ratio": 0.3      # Min appearance ratio (recommended: 0.2-0.4)
    }
}

def load_settings(path):
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if path and Path(path).exists():
        user = json.loads(Path(path).read_text())
        # shallow-merge top levels
        for k in user:
            if isinstance(user[k], dict):
                cfg[k].update(user[k])
            else:
                cfg[k] = user[k]
    return cfg

# ---------- Camera config (live only) ----------
def configure_camera(cap, args):
    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    if args.cam_w>0: cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.cam_w)
    if args.cam_h>0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)
    if args.cam_fps>0: cap.set(cv2.CAP_PROP_FPS, args.cam_fps)
    if args.auto_exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if args.auto_exposure else 0.25)
    if args.exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(args.exposure))
    if args.auto_wb is not None:
        cap.set(cv2.CAP_PROP_AUTO_WB, 1 if args.auto_wb else 0)
    if args.wb_temp is not None:
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, int(args.wb_temp))
    if args.brightness is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, float(args.brightness))
    if args.gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, float(args.gain))
    if args.auto_focus is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if args.auto_focus else 0)
    if args.focus is not None:
        cap.set(cv2.CAP_PROP_FOCUS, float(args.focus))

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="0", help="Camera index or path to video/image")
    ap.add_argument("--settings", default="", help="JSON with warp/preprocess/blob (applied to ALL sources)")

    # camera (live only)
    ap.add_argument("--backend", default="", help="DSHOW/MSMF")
    ap.add_argument("--fourcc",  default="", help="e.g. MJPG, YUY2")
    ap.add_argument("--cam-w", type=int, default=1280)
    ap.add_argument("--cam-h", type=int, default=720)
    ap.add_argument("--cam-fps", type=int, default=30)
    ap.add_argument("--auto-exposure", type=lambda s:s.lower() in ("1","true","yes"), default=None)
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument("--auto-wb", type=lambda s:s.lower() in ("1","true","yes"), default=None)
    ap.add_argument("--wb-temp", type=int, default=None)
    ap.add_argument("--brightness", type=float, default=None)
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--auto-focus", type=lambda s:s.lower() in ("1","true","yes"), default=None)
    ap.add_argument("--focus", type=float, default=None)

    # quick overrides (optional)
    ap.add_argument("--zoom", type=float, default=None)
    ap.add_argument("--rotate", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)

    # temporal aggregation
    ap.add_argument("--no-temporal", action="store_true", help="Disable temporal aggregation")
    ap.add_argument("--temporal-frames", type=int, default=None, help="Number of frames to track (default: 30)")
    ap.add_argument("--temporal-distance", type=float, default=None, help="Cluster distance in pixels (default: 30.0)")
    ap.add_argument("--temporal-ratio", type=float, default=None, help="Min detection ratio 0.0-1.0 (default: 0.3)")

    # show FPS
    ap.add_argument("--show-fps", action="store_true")
    args = ap.parse_args()

    cfg = load_settings(args.settings)
    # overrides
    if args.zoom   is not None: cfg["preprocess"]["zoom"] = args.zoom
    if args.rotate is not None: cfg["preprocess"]["rotate_deg"] = args.rotate
    if args.alpha  is not None: cfg["preprocess"]["alpha"] = args.alpha
    if args.beta   is not None: cfg["preprocess"]["beta"]  = args.beta
    if args.gamma  is not None: cfg["preprocess"]["gamma"] = args.gamma
    
    # temporal overrides
    if args.no_temporal:
        cfg["temporal"]["enabled"] = False
    if args.temporal_frames is not None:
        cfg["temporal"]["max_frames"] = args.temporal_frames
    if args.temporal_distance is not None:
        cfg["temporal"]["cluster_distance"] = args.temporal_distance
    if args.temporal_ratio is not None:
        cfg["temporal"]["min_detection_ratio"] = args.temporal_ratio

    warp_w = cfg["warp"]["width"]; warp_h = cfg["warp"]["height"]
    M = None
    if cfg["warp"]["quad"]:
        M = load_quad(cfg["warp"]["quad"], warp_w, warp_h)

    # single image → one shot
    src_arg = args.src
    is_image = isinstance(src_arg, str) and src_arg.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff"))
    detector = build_blob_params(cfg["blob"])
    
    # Initialize temporal tracker (only for video/camera streams)
    tracker = None
    if not is_image and cfg["temporal"]["enabled"]:
        # Validate max_frames - if 0 or negative, disable temporal tracking
        if cfg["temporal"]["max_frames"] <= 0:
            print("WARNING: max_frames <= 0, disabling temporal tracking")
            cfg["temporal"]["enabled"] = False
        else:
            tracker = TemporalPillTracker(
                max_frames=cfg["temporal"]["max_frames"],
                cluster_distance=cfg["temporal"]["cluster_distance"],
                min_detection_ratio=cfg["temporal"]["min_detection_ratio"]
            )
            print(f"Temporal tracking: {cfg['temporal']['max_frames']} frames, "
                  f"distance={cfg['temporal']['cluster_distance']:.1f}px, "
                  f"ratio={cfg['temporal']['min_detection_ratio']:.2f}")

    def process_frame(frame, use_temporal=True):
        # unified preprocessing (identical for camera & video)
        color, gray = preprocess_frame(frame, cfg["preprocess"])
        warped = warp_topdown(color, M, (warp_w, warp_h)) if M is not None else cv2.resize(color, (warp_w, warp_h))
        gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # CLAHE/blur again after warp to stabilize size/contrast
        clahe = cv2.createCLAHE(clipLimit=cfg["preprocess"]["clahe_clip"], tileGridSize=(cfg["preprocess"]["clahe_tile"], cfg["preprocess"]["clahe_tile"]))
        gray_w = clahe.apply(gray_w)
        if cfg["preprocess"]["blur"] > 0:
            k = cfg["preprocess"]["blur"] + (1 - cfg["preprocess"]["blur"] % 2)
            gray_w = cv2.GaussianBlur(gray_w, (k, k), 0)
        kps = detector.detect(gray_w)
        
        # Get count with temporal aggregation if enabled
        if use_temporal and tracker is not None:
            tracker.add_frame(kps)
            stable_count = tracker.get_stable_count()
            current_count = len(kps)
            count_text = f"stable={stable_count} (current={current_count})"
            # Color coding based on stable count
            if stable_count >= 8:
                count_color = (0, 255, 0)  # Green - OK
            elif stable_count >= 6:
                count_color = (0, 165, 255)  # Orange - partial detection
            else:
                count_color = (0, 0, 255)  # Red - likely defect
        else:
            stable_count = len(kps)
            count_text = f"count={stable_count}"
            count_color = (0, 255, 0)
        
        vis = cv2.drawKeypoints(warped, kps, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.putText(vis, count_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, count_color, 2)
        return vis

    if is_image:
        img = cv2.imread(src_arg)
        if img is None: print(f"ERROR: cannot open image {src_arg}"); return
        out = process_frame(img, use_temporal=False)
        cv2.imshow("pill_count_warp", out)
        cv2.waitKey(0); cv2.destroyAllWindows()
        return

    # open camera/video
    # (backend hint for Windows)
    if args.backend.upper() == "DSHOW":
        cap = cv2.VideoCapture(int(src_arg) if src_arg.isdigit() else src_arg, cv2.CAP_DSHOW)
    elif args.backend.upper() == "MSMF":
        cap = cv2.VideoCapture(int(src_arg) if src_arg.isdigit() else src_arg, cv2.CAP_MSMF)
    else:
        cap = cv2.VideoCapture(int(src_arg) if src_arg.isdigit() else src_arg)

    if not cap.isOpened():
        print(f"ERROR: cannot open source {src_arg}"); return

    # Live-only properties (videos ignore these; that’s fine — our processing is identical)
    if src_arg.isdigit():
        configure_camera(cap, args)

    last, frames, fps_txt = time.time(), 0, ""
    while True:
        ok, frame = cap.read()
        if not ok: break
        vis = process_frame(frame, use_temporal=(tracker is not None))
        if args.show_fps:
            frames += 1
            now = time.time()
            if now - last >= 0.5:
                fps_txt = f"{frames/(now-last):.1f} FPS"; frames = 0; last = now
            cv2.putText(vis, fps_txt, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.imshow("pill_count_warp", vis)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
