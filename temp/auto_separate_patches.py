import os, shutil, argparse
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy as shannon_entropy

ROOT = Path("patch_data")  # contains train/ val/ test/

# ---------- Feature + helpers ----------
def features(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(g.mean())
    std  = float(g.std())
    lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())

    med = np.median(g)
    lower = max(0, 0.66 * med)
    upper = min(255, 1.33 * med)
    edges = cv2.Canny(g, lower, upper)
    edge_ratio = float(edges.mean()) / 255.0

    hist = cv2.calcHist([g],[0],None,[64],[0,256]).ravel()
    hist = hist / (hist.sum() + 1e-8)
    ent = float(shannon_entropy(hist, base=2))

    return np.array([edge_ratio, lap_var, std, mean, ent], dtype=np.float32)

def is_black(img_bgr, mean_thr=12, dark_pct_thr=0.85, dark_val=18):
    """Return True if patch is near-black (idle capture / no scene)."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean = g.mean()
    dark_pct = (g < dark_val).mean()
    return (mean < mean_thr) or (dark_pct > dark_pct_thr)

def parse_cell_index(name: str):
    # expects ..._c{i}_....jpg → return i or -1 if missing
    try:
        base = Path(name).stem
        for p in base.split("_"):
            if p.startswith("c") and p[1:].isdigit():
                return int(p[1:])
    except:
        pass
    return -1

def cluster_and_decide(X, invert_rule=False):
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    c0 = X[labels==0].mean(axis=0)  # [edge, lap, std, mean, ent]
    c1 = X[labels==1].mean(axis=0)

    # Texture score (higher => more structure)
    s0 = c0[0] + c0[1] + c0[2]
    s1 = c1[0] + c1[1] + c1[2]

    # normal: lower score => empty ; invert: higher score => empty
    empty_label = (0 if s0 < s1 else 1) if not invert_rule else (0 if s0 > s1 else 1)
    return labels, empty_label, (c0, c1, (s0, s1))

# ---------- Core ----------
def process_split(split_dir: Path, per_cell: bool, dry_run: bool,
                  invert_rule: bool, auto_flip_threshold: float,
                  black_first: bool, delete_black: bool,
                  min_cluster_size: int,
                  black_mean_thr: int, black_dark_pct_thr: float, black_dark_val: int):
    present_dir = split_dir / "present"
    empty_dir   = split_dir / "empty"
    empty_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(present_dir.glob("*.jpg"))
    if not files:
        return 0, 0

    moved = 0
    total = 0

    if per_cell:
        groups = {}
        for f in files:
            ci = parse_cell_index(f.name)
            groups.setdefault(ci, []).append(f)

        for ci, group in groups.items():
            total += len(group)
            # Step 1: black-first filtering
            pool_feats, pool_files = [], []
            if black_first:
                for f in group:
                    img = cv2.imread(str(f))
                    if img is None: continue
                    if is_black(img, mean_thr=black_mean_thr,
                                dark_pct_thr=black_dark_pct_thr,
                                dark_val=black_dark_val):
                        if not dry_run:
                            if delete_black:
                                os.remove(f)
                            else:
                                shutil.move(str(f), str(empty_dir / f.name))
                        moved += (0 if dry_run else 1)
                    else:
                        pool_feats.append(features(img))
                        pool_files.append(f)
            else:
                for f in group:
                    img = cv2.imread(str(f))
                    if img is None: continue
                    pool_feats.append(features(img))
                    pool_files.append(f)

            if len(pool_files) < min_cluster_size:
                continue

            X = np.vstack(pool_feats)
            labels, empty_label, stats = cluster_and_decide(X, invert_rule=invert_rule)

            # Auto-flip if absurd ratio
            empty_ratio = float((labels == empty_label).sum()) / len(labels)
            if auto_flip_threshold and empty_ratio > auto_flip_threshold:
                empty_label = 1 - empty_label
                empty_ratio = float((labels == empty_label).sum()) / len(labels)

            if dry_run:
                print(f"[DRY] {split_dir.name} c{ci}: empties {int(empty_ratio*len(labels))}/{len(labels)} "
                      f"(invert={invert_rule})")
            else:
                for f, lab in zip(pool_files, labels):
                    if lab == empty_label:
                        shutil.move(str(f), str(empty_dir / f.name))
                        moved += 1

    else:
        # all patches together
        pool_feats, pool_files = [], []
        for f in files:
            img = cv2.imread(str(f))
            if img is None: continue
            if black_first and is_black(img, mean_thr=black_mean_thr,
                                        dark_pct_thr=black_dark_pct_thr,
                                        dark_val=black_dark_val):
                if not dry_run:
                    if delete_black:
                        os.remove(f)
                    else:
                        shutil.move(str(f), str(empty_dir / f.name))
                moved += (0 if dry_run else 1)
            else:
                pool_feats.append(features(img))
                pool_files.append(f)

        total += len(files)
        if len(pool_files) >= min_cluster_size:
            X = np.vstack(pool_feats)
            labels, empty_label, stats = cluster_and_decide(X, invert_rule=invert_rule)
            empty_ratio = float((labels == empty_label).sum()) / len(labels)
            if auto_flip_threshold and empty_ratio > auto_flip_threshold:
                empty_label = 1 - empty_label

            if dry_run:
                print(f"[DRY] {split_dir.name}: empties {int((labels==empty_label).sum())}/{len(pool_files)} "
                      f"(invert={invert_rule})")
            else:
                for f, lab in zip(pool_files, labels):
                    if lab == empty_label:
                        shutil.move(str(f), str(empty_dir / f.name))
                        moved += 1

    return total, moved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", action="store_true", help="Cluster per cell index (recommended).")
    ap.add_argument("--dry-run", action="store_true", help="Only print counts; do not move files.")
    ap.add_argument("--invert-rule", action="store_true", help="Treat higher-edge/texture cluster as EMPTY.")
    ap.add_argument("--auto-flip-thr", type=float, default=0.60,
                    help="If chosen empty cluster > this ratio, flip decision (0 disables).")
    ap.add_argument("--black-first", action="store_true",
                    help="Move near-black patches to empty before clustering.")
    ap.add_argument("--delete-black", action="store_true",
                    help="Delete near-black patches instead of moving to empty.")
    ap.add_argument("--min-cluster-size", type=int, default=8, help="Min samples to run KMeans.")
    ap.add_argument("--black-mean-thr", type=int, default=12, help="Mean gray threshold for black test.")
    ap.add_argument("--black-dark-pct-thr", type=float, default=0.85,
                    help="Fraction of pixels below dark_val to be considered black.")
    ap.add_argument("--black-dark-val", type=int, default=18,
                    help="Pixel value defining 'very dark' for black test.")
    ap.add_argument("--root", default=str(ROOT), help="patch_data root.")
    args = ap.parse_args()

    root = Path(args.root)
    totals = moved = 0
    for split in ["train","val","test"]:
        t, m = process_split(
            root / split,
            per_cell=args.per_cell,
            dry_run=args.dry_run,
            invert_rule=args.invert_rule,
            auto_flip_threshold=args.auto_flip_thr,
            black_first=args.black_first,
            delete_black=args.delete_black,
            min_cluster_size=args.min_cluster_size,
            black_mean_thr=args.black_mean_thr,
            black_dark_pct_thr=args.black_dark_pct_thr,
            black_dark_val=args.black_dark_val,
        )
        totals += t
        moved  += m

    if args.dry_run:
        print(f"[DRY] Total examined: {totals}")
    else:
        print(f"Moved {moved} patches from present → empty out of {totals} examined.")

if __name__ == "__main__":
    main()
