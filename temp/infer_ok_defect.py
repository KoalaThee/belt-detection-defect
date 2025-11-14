# infer_ok_defect.py
import cv2, json, time, argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# ---- If you have your own split_grid, you can import it; otherwise use this ----
def split_grid(img_bgr, rows=2, cols=3, pad=4):
    H, W = img_bgr.shape[:2]
    cell_w = W // cols
    cell_h = H // rows
    cells, boxes = [], []
    for r in range(rows):
        for c in range(cols):
            x1 = max(c*cell_w + pad, 0);   x2 = min((c+1)*cell_w - pad, W)
            y1 = max(r*cell_h + pad, 0);   y2 = min((r+1)*cell_h - pad, H)
            boxes.append((x1, y1, x2, y2))
            cells.append(img_bgr[y1:y2, x1:x2])
    return cells, boxes

def load_quad(json_path, warp_w, warp_h):
    with open(json_path, "r") as f:
        pts = np.array(json.load(f)["points"], dtype=np.float32)  # TL,TR,BR,BL
    dst = np.array([[0,0],[warp_w-1,0],[warp_w-1,warp_h-1],[0,warp_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return M

def warp_topdown(bgr, M, size):
    return cv2.warpPerspective(bgr, M, size)

# -------------------- Inference config --------------------
ROWS, COLS = 2, 3
IMG = (96, 96)              # training input size
EMPTY_IDX = 0               # class_names were ['empty','present'] → empty at index 0
THR = 0.85                  # chosen from sweep to reduce tray false alarms
K_CONSEC = 0                # temporal smoothing (0 = disabled, try 2–3 to enable)
MIN_EMPTY_FOR_DEFECT = 1    # tray rule: defect if ≥ this many cells are empty
WARP_W, WARP_H = 600, 300   # warped board size (match your patch builder)

# -------------------- Load model once ---------------------
MODEL = tf.keras.models.load_model("models/pill_present_empty.keras")

def prob_empty_from_patch(patch_bgr):
    """Resize to 96x96 and get P(empty). Do NOT /255; model already rescaled inside."""
    x = cv2.resize(patch_bgr, IMG).astype(np.float32)  # keep 0..255; Rescaling is inside the model
    p = MODEL(x[None], training=False).numpy()[0]      # [p_empty, p_present]
    return float(p[EMPTY_IDX])

def classify_warped_frame(warped_bgr):
    """Return per-cell decisions using THR on P(empty)."""
    cells, boxes = split_grid(warped_bgr, ROWS, COLS)
    p_empty = [prob_empty_from_patch(c) for c in cells]
    is_empty_now = [pe >= THR for pe in p_empty]       # thresholded decision per cell
    return is_empty_now, p_empty, cells, boxes

def overlay(warped_bgr, boxes, p_empty, decided_empty, fps=None):
    vis = warped_bgr.copy()
    for i, ((x1,y1,x2,y2), pe, de) in enumerate(zip(boxes, p_empty, decided_empty)):
        color = (0,0,255) if de else (0,200,0)  # red=empty(decided), green=present
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        txt = f"pe={pe:.2f}"
        cv2.putText(vis, txt, (x1+4, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(vis, f"c{i}", (x1+4, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    empty_count = sum(decided_empty)
    label = "DEFECT" if empty_count >= MIN_EMPTY_FOR_DEFECT else "OK"
    color = (0,0,255) if label=="DEFECT" else (0,200,0)
    cv2.putText(vis, f"{label}  empty={empty_count}/{ROWS*COLS}  THR={THR:.2f}  K={K_CONSEC}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if fps is not None:
        cv2.putText(vis, f"{fps:.1f} FPS", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    return vis

def run_stream(src=0, quad_json=None):
    # Optional perspective warp
    M = None
    if quad_json:
        M = load_quad(quad_json, WARP_W, WARP_H)

    # Temporal smoothing: keep per-cell streaks if enabled
    empty_streak = [0]*(ROWS*COLS)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: cannot open source {src}")
        return

    t0, frames = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        if M is not None:
            warped = warp_topdown(frame, M, (WARP_W, WARP_H))
        else:
            # Fallback: simple resize (not ideal—use quad if you have it)
            warped = cv2.resize(frame, (WARP_W, WARP_H))

        is_empty_now, p_empty, cells, boxes = classify_warped_frame(warped)

        # Apply temporal smoothing if requested (require K_CONSEC consecutive empties)
        if K_CONSEC > 0:
            decided_empty = []
            for i, e_now in enumerate(is_empty_now):
                empty_streak[i] = empty_streak[i] + 1 if e_now else 0
                decided_empty.append(empty_streak[i] >= K_CONSEC)
        else:
            decided_empty = is_empty_now

        # FPS calc
        if frames % 10 == 0:
            t1 = time.time()
            fps = 10.0 / max(1e-6, (t1 - t0))
            t0 = t1
        # draw
        vis = overlay(warped, boxes, p_empty, decided_empty, fps=None)

        cv2.imshow("blister_infer", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="0",
                    help="Camera index (e.g., 0) or video path (e.g., clips/OK/sample.mp4)")
    ap.add_argument("--quad", default="", help="Path to config/quad.json for perspective warp (optional)")
    ap.add_argument("--thr", type=float, default=THR, help="Probability threshold for EMPTY (p_empty >= thr)")
    ap.add_argument("--k", type=int, default=K_CONSEC, help="Require K consecutive empty frames per cell (0=off)")
    ap.add_argument("--min-empty", type=int, default=MIN_EMPTY_FOR_DEFECT,
                    help="Tray is DEFECT if >= this many cells are empty")
    args = ap.parse_args()

    # override globals from CLI
    THR = args.thr
    K_CONSEC = args.k
    MIN_EMPTY_FOR_DEFECT = args.min_empty

    # parse source
    src = int(args.src) if args.src.isdigit() else args.src
    quad_json = args.quad if args.quad else None

    run_stream(src=src, quad_json=quad_json)
