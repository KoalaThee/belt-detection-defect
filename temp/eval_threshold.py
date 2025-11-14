# eval_thresholded.py
import numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import image_dataset_from_directory

IMG=(96,96)
TEST_ROOT="patch_data/test"
MODEL="models/pill_present_empty.keras"
EMPTY_IDX=0
THR=0.85  # ← set from sweep

ds = image_dataset_from_directory(TEST_ROOT, image_size=IMG, batch_size=512, label_mode="int", shuffle=False)
class_names = ds.class_names
assert class_names[EMPTY_IDX] == "empty", class_names

model = tf.keras.models.load_model(MODEL)
y_true, y_pred = [], []
for x, y in ds:
    p = model(x, training=False).numpy()
    pe = p[:, EMPTY_IDX]
    pred_cls = np.where(pe >= THR, EMPTY_IDX, 1 - EMPTY_IDX)
    y_pred.append(pred_cls); y_true.append(y.numpy())
y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)

print("Class order:", class_names)
print("\nClassification report (thresholded):")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_true, y_pred))
