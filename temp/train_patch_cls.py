# train_patch_cls.py
import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(42)

IMG = (96, 96)
BATCH = 64
EPOCHS = 15
TRAIN_ROOT = "patch_data/train"
VAL_ROOT   = "patch_data/val"
MODEL_OUT  = "models/pill_present_empty.keras"

Path("models").mkdir(parents=True, exist_ok=True)

def make_ds(root, shuffle=True):
    # integer labels so class_weight works
    ds = tf.keras.utils.image_dataset_from_directory(
        root,
        image_size=IMG,
        batch_size=BATCH,
        label_mode="int",
        shuffle=shuffle,
    )
    return ds.prefetch(tf.data.AUTOTUNE)

def count_files(root):
    root = Path(root)
    counts = {}
    for cls in sorted([p.name for p in root.iterdir() if p.is_dir()]):
        counts[cls] = sum(1 for _ in (root / cls).glob("*.jpg"))
    return counts

# datasets
train_ds = make_ds(TRAIN_ROOT, shuffle=True)
val_ds   = make_ds(VAL_ROOT,   shuffle=False)

# class order alphabetical, e.g. ['empty','present']
train_counts = count_files(TRAIN_ROOT)
classes_sorted = sorted(train_counts.keys())
assert len(classes_sorted) == 2, f"Expected 2 classes, got: {classes_sorted}"
class_to_index = {cls: i for i, cls in enumerate(classes_sorted)}
print("Classes (alphabetical):", classes_sorted)
print("Train counts:", train_counts)

total = sum(train_counts.values())
class_weight = {
    class_to_index[cls]: total / (2.0 * max(1, n))
    for cls, n in train_counts.items()
}
print("Class weights (index->weight):", class_weight)

# model
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomContrast(0.10),
    layers.RandomBrightness(0.15),
], name="aug")

inputs = keras.Input(shape=(*IMG, 3))
x = data_aug(inputs)
x = layers.Rescaling(1./255)(x)
x = layers.Conv2D(32, 3, activation="relu")(x); x = layers.MaxPool2D()(x)
x = layers.Conv2D(64, 3, activation="relu")(x); x = layers.MaxPool2D()(x)
x = layers.Conv2D(128, 3, activation="relu")(x); x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs, out)

# use sparse loss + sparse accuracy only (avoid shape conflicts)
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

cb = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_acc", mode="max"),
    keras.callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_acc", mode="max", save_best_only=True),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cb,
    class_weight=class_weight,
)

model.save(MODEL_OUT)
print(f"Saved model to {MODEL_OUT}")
