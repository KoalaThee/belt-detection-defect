# move_empty_patches.py
import cv2, shutil
from pathlib import Path

ROOT = Path("patch_data")   # contains train/ val/ test/
EDGE_RATIO_THR = 0.004     # lower = stricter empty
STDDEV_THR     = 5.0        # lower = stricter empty
MEAN_THR       = 50         # very dark + low var => empty

def is_empty(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)                      # stabilize contrast
    edges = cv2.Canny(g, 50, 150)
    edge_ratio = edges.mean() / 255.0            # 0..1
    stddev = g.std()
    mean   = g.mean()
    # empty if few edges & low texture, or very dark & low texture
    if edge_ratio < EDGE_RATIO_THR and stddev < STDDEV_THR:
        return True
    if mean < MEAN_THR and stddev < STDDEV_THR * 0.9:
        return True
    return False

moved = 0
for split in ["train","val","test"]:
    present_dir = ROOT / split / "present"
    empty_dir   = ROOT / split / "empty"
    if not present_dir.exists(): continue
    empty_dir.mkdir(parents=True, exist_ok=True)

    for jpg in list(present_dir.glob("*.jpg")):
        img = cv2.imread(str(jpg))
        if img is None: continue
        if is_empty(img):
            shutil.move(str(jpg), str(empty_dir / jpg.name))
            moved += 1

print(f"Done. Moved {moved} likely-empty patches to patch_data/*/empty/")
