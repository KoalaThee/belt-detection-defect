# sweep_threshold_fast.py
import numpy as np, tensorflow as tf, sys
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.utils import image_dataset_from_directory

IMG = (96, 96)
VAL_ROOT = "patch_data/val"                  # use VAL to choose threshold
MODEL    = "models/pill_present_empty.keras"
EMPTY_IDX = 0                                # alphabetical -> ['empty','present']
N_CELLS = 6                                  # pills per tray for tray-FP estimate
BATCH = 512

def tray_fp_from_cell_fp(cell_fp, n=N_CELLS):
    return 1 - (1 - cell_fp)**n

print("Loading dataset…", flush=True)
ds = image_dataset_from_directory(
    VAL_ROOT, image_size=IMG, batch_size=BATCH, label_mode="int", shuffle=False
)
class_names = ds.class_names
assert class_names[EMPTY_IDX] == "empty", f"class order is {class_names}, expected empty first"

print("Loading model…", flush=True)
model = tf.keras.models.load_model(MODEL)

print("Computing probabilities…", flush=True)
y_true_list, p_empty_list = [], []
for x, y in ds:
    p = model(x, training=False).numpy()
    y_true_list.append(y.numpy())
    p_empty_list.append(p[:, EMPTY_IDX])
y_true = np.concatenate(y_true_list)
p_empty = np.concatenate(p_empty_list)

print(f"VAL samples: {len(y_true)}  |  empty={np.sum(y_true==EMPTY_IDX)}  present={np.sum(y_true!=EMPTY_IDX)}")
print(f"p_empty stats → min={p_empty.min():.3f}  p25={np.percentile(p_empty,25):.3f}  "
      f"med={np.median(p_empty):.3f}  p75={np.percentile(p_empty,75):.3f}  max={p_empty.max():.3f}\n")

print(" thr |  Prec_e  Rec_e  F1_e | cellFP -> trayFP(6)")
best_rows = []  # store (thr, prec, rec, f1, cell_fp, tray_fp)

for thr in np.linspace(0.50, 0.95, 10):
    pred_empty = (p_empty >= thr).astype(int)                # 1=empty (binary view)
    true_empty = (y_true == EMPTY_IDX).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(true_empty, pred_empty, average="binary", zero_division=0)
    # per-cell FP := P(present predicted empty)
    present_mask = (y_true != EMPTY_IDX)
    cell_fp = ( (present_mask) & (pred_empty == 1) ).sum() / present_mask.sum()
    tray_fp = tray_fp_from_cell_fp(cell_fp)

    print(f"{thr:5.2f} |   {prec:6.3f} {rec:6.3f} {f1:6.3f} |  {cell_fp:6.3f} -> {tray_fp:6.3f}")
    best_rows.append((thr, prec, rec, f1, cell_fp, tray_fp))

# simple recommendations by tray-FP targets
def pick_target(target):
    # choose highest recall among thresholds with tray_fp <= target
    candidates = [r for r in best_rows if r[5] <= target]
    return max(candidates, key=lambda r: r[2]) if candidates else None

print("\nRecommended thresholds by tray-FP target:")
for tgt in [0.15, 0.10, 0.05]:
    row = pick_target(tgt)
    if row:
        thr, prec, rec, f1, cfp, tfp = row
        print(f"  target {tgt*100:>2.0f}% → THR={thr:.2f} | Prec_e={prec:.3f} Rec_e={rec:.3f} F1_e={f1:.3f} | cellFP={cfp:.3f} trayFP={tfp:.3f}")
    else:
        print(f"  target {tgt*100:>2.0f}% → No threshold meets this (model too trigger-happy).")
