
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset path
DATASET_DIR = "dataset"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

# Check dataset folders
mask_path = os.path.join(DATASET_DIR, "with_mask")
no_mask_path = os.path.join(DATASET_DIR, "without_mask")

if not os.path.isdir(mask_path) or not os.path.isdir(no_mask_path):
    raise FileNotFoundError(
        "Create dataset/with_mask and dataset/without_mask "
        "and place images inside them."
    )

# Load and preprocess images
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    classes=["with_mask", "without_mask"],
    seed=42
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    classes=["with_mask", "without_mask"],
    seed=42,
    shuffle=False
)

if train_data.samples == 0 or val_data.samples == 0:
    raise ValueError("No images found. Check your dataset folders.")

print("Class labels:", train_data.class_indices)

# Build the CNN model
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train the model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save the trained model
os.makedirs("model", exist_ok=True)
model.save("model/mask_detector.keras")

print("Training completed!")
print("Model saved to model/mask_detector.keras")
