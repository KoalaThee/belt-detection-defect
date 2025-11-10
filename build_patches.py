# build_patches.py
import cv2, os, json, glob, random
import numpy as np
from pathlib import Path
random.seed(42)

ROWS, COLS = 2, 3
WARP_W, WARP_H = 600, 300
PATCH = (96,96)
KEEP_EVERY = 5        # take every 5th frame
SPLIT = {"train":0.7, "val":0.15, "test":0.15}

SRC = Path("data")
OUT = Path("patch_data")
OUT.mkdir(exist_ok=True)

with open("config/quad.json") as f:
    P = np.array(json.load(f)["points"], dtype=np.float32)  # TL,TR,BR,BL
dst = np.array([[0,0],[WARP_W-1,0],[WARP_W-1,WARP_H-1],[0,WARP_H-1]], dtype=np.float32)
M = cv2.getPerspectiveTransform(P, dst)

for split in ["train","val","test"]:
    for cls in ["present","empty"]:
        (OUT/split/cls).mkdir(parents=True, exist_ok=True)

def split_grid(img, rows=ROWS, cols=COLS, pad=4):
    H,W = img.shape[:2]; h=W//cols; v=H//rows
    patches=[]
    for r in range(rows):
        for c in range(cols):
            x1=max(c*h+pad,0); x2=min((c+1)*h-pad,W)
            y1=max(r*v+pad,0); y2=min((r+1)*v-pad,H)
            patches.append(img[y1:y2, x1:x2])
    return patches

def choose_split():
    r=random.random()
    return "train" if r<SPLIT["train"] else "val" if r<SPLIT["train"]+SPLIT["val"] else "test"

def process_video(path:Path):
    cap=cv2.VideoCapture(str(path))
    idx=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if idx%KEEP_EVERY!=0: idx+=1; continue

        warped=cv2.warpPerspective(frame,M,(WARP_W,WARP_H))
        cells=split_grid(warped)
        s=choose_split()
        for i,cell in enumerate(cells):
            patch=cv2.resize(cell,PATCH)
            # default = assume present, quick to relabel later
            cv2.imwrite(str(OUT/s/"present"/f"{path.stem}_c{i}_{idx:06d}.jpg"), patch)
        idx+=1
    cap.release()

def main():
    vids = list(SRC.glob("OK/*.mp4")) + list(SRC.glob("Defect/*.mp4"))
    print(f"Found {len(vids)} videos")
    for v in vids: process_video(v)
    print("Done. Now quickly relabel empties in patch_data/*/present → patch_data/*/empty")

if __name__=="__main__":
    main()
