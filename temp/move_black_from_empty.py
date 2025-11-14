# move_black_from_empty.py
import argparse, shutil
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("patch_data")          # has train/ val/ test/
OUT  = Path("patch_data/_black")   # quarantine destination

def is_black(img_bgr, mean_thr=12, dark_pct_thr=0.85, dark_val=18):
    """Return True if patch is near-black (idle/no-tray)."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean = g.mean()
    dark_pct = (g < dark_val).mean()
    return (mean < mean_thr) or (dark_pct > dark_pct_thr)

def move_black_from_split(split_dir: Path, out_dir: Path, dry: bool,
                          mean_thr: int, dark_pct_thr: float, dark_val: int):
    src = split_dir / "empty"
    if not src.exists():
        return 0, 0
    dst = out_dir / split_dir.name / "empty_black"
    dst.mkdir(parents=True, exist_ok=True)

    checked = moved = 0
    for p in src.glob("*.jpg"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        checked += 1
        if is_black(img, mean_thr=mean_thr, dark_pct_thr=dark_pct_thr, dark_val=dark_val):
            if dry:
                print(f"[DRY] would move: {p} -> {dst/p.name}")
            else:
                shutil.move(str(p), str(dst / p.name))
            moved += 1
    return checked, moved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT), help="Root with train/val/test (default: patch_data)")
    ap.add_argument("--out",  default=str(OUT),  help="Quarantine folder (default: patch_data/_black)")
    ap.add_argument("--dry-run", action="store_true", help="Only print actions; do not move files")
    ap.add_argument("--mean-thr", type=int, default=12, help="Mean gray threshold for black test")
    ap.add_argument("--dark-pct-thr", type=float, default=0.85,
                    help="Fraction of pixels below dark_val to be considered black")
    ap.add_argument("--dark-val", type=int, default=18, help="Pixel value that counts as 'very dark'")
    args = ap.parse_args()

    root = Path(args.root)
    out  = Path(args.out)

    total_checked = total_moved = 0
    for split in ["train", "val", "test"]:
        checked, moved = move_black_from_split(
            root / split, out, args.dry_run, args.mean_thr, args.dark_pct_thr, args.dark_val
        )
        print(f"{split:>5}: checked {checked}, moved {moved}")
        total_checked += checked
        total_moved   += moved

    if args.dry_run:
        print(f"[DRY] Total checked: {total_checked}, would move: {total_moved}")
    else:
        print(f"Total checked: {total_checked}, moved: {total_moved}")

if __name__ == "__main__":
    main()
