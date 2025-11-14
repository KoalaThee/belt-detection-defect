# eval_patch_cls.py (threshold-aware)
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

IMG=(96,96)
ROOT="patch_data/test"
MODEL="models/pill_present_empty.keras"
USE_THRESHOLD=True            # ← set True to use custom threshold
EMPTY_IDX=0                   # class_names are alphabetical -> ['empty','present']
THR=0.85                      # ← paste the chosen threshold from the sweep

ds=tf.keras.utils.image_dataset_from_directory(
    ROOT, image_size=IMG, batch_size=64, label_mode="int", shuffle=False)

class_names=ds.class_names
assert class_names[EMPTY_IDX]=="empty", class_names

model=tf.keras.models.load_model(MODEL)

y_true=[]; y_pred=[]
for x,y in ds:
    p=model(x,training=False).numpy()          # (N,2)
    if USE_THRESHOLD:
        pe=p[:,EMPTY_IDX]
        pred=np.where(pe>=THR, EMPTY_IDX, 1-EMPTY_IDX)
    else:
        pred=p.argmax(1)
    y_true.append(y.numpy())
    y_pred.append(pred)

y_true=np.concatenate(y_true)
y_pred=np.concatenate(y_pred)

mode = f"thresholded (THR={THR})" if USE_THRESHOLD else "argmax (0.5 default)"
print("Class order:", class_names)
print(f"\nClassification report — {mode}:")
print(classification_report(y_true,y_pred,target_names=class_names,digits=4))
print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_true,y_pred))
