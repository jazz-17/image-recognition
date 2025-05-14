import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.applications import MobileNetV2 # A good pre-trained model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil # For creating example data

# --- Configuration ---
DATA_DIR = 'dataset' # Change this to your dataset directory
IMG_WIDTH, IMG_HEIGHT = 160, 160 # Input image size for MobileNetV2
BATCH_SIZE = 32
EPOCHS = 50 # Adjust as needed, EarlyStopping will help
VAL_SPLIT = 0.2 # 20% of data for validation
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'laptop_tablet_phone_classifier.keras' # Keras's new format

# --- 0. (Optional) Create Dummy Data if you don't have it ---
def create_dummy_data(base_dir, num_images_per_class=10):
    if os.path.exists(base_dir):
        print(f"Directory '{base_dir}' already exists. Skipping dummy data creation.")
        if not any(os.scandir(base_dir)): # Check if it's empty
             print(f"'{base_dir}' is empty. You might want to delete it and rerun if you expect dummy data.")
        return

    print(f"Creating dummy data in '{base_dir}'...")
    os.makedirs(base_dir, exist_ok=True)
    classes = ['laptop', 'tablet', 'phone']
    for class_name in classes:
        class_path = os.path.join(base_dir, class_name)
        os.makedirs(class_path, exist_ok=True)
        for i in range(num_images_per_class):
            # Create a tiny random image (3 channels for RGB)
            dummy_img_array = np.random.randint(0, 256, size=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            img = tf.keras.preprocessing.image.array_to_img(dummy_img_array)
            img.save(os.path.join(class_path, f"{class_name}_dummy_{i+1}.png"))
    print("Dummy data created.")

# Create dummy data if DATA_DIR does not exist or is empty
if not os.path.exists(DATA_DIR) or not any(os.scandir(DATA_DIR)):
    create_dummy_data(DATA_DIR, num_images_per_class=50) # Create 50 dummy images per class
    print(f"\nINFO: Dummy data was created in '{DATA_DIR}'. Replace this with your actual image data.\n")
elif not all(os.path.exists(os.path.join(DATA_DIR, c)) for c in ['laptop', 'tablet', 'phone']):
    print(f"WARNING: Expected subdirectories 'laptop', 'tablet', 'phone' in '{DATA_DIR}' not found.")
    print("Please ensure your data is structured correctly or run with an empty/non-existent DATA_DIR to create dummy data.")
    # exit() # You might want to exit if data structure is wrong

# --- 1. Load and Preprocess Data ---
print("Loading and preprocessing data...")

# Keras utility for loading images from directories
# It can automatically infer class labels from the subdirectory names
# and split data into training and validation sets.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=123, # Shuffle seed for reproducibility
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical' # For 'categorical_crossentropy' loss
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found classes: {class_names}")
print(f"Number of classes: {num_classes}")

if num_classes < 2:
    print("Error: Not enough classes found. Need at least 2. Check your DATA_DIR structure.")
    exit()

# --- 2. Data Augmentation and Preprocessing Layers ---
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
])

# Normalize pixel values. For MobileNetV2, it's often better to use its own preprocess_input
# but for simplicity here, we'll scale to [0,1] and then the base model will handle its specific needs if any.
# If using tf.keras.applications.mobilenet_v2.preprocess_input, apply it after loading images.
# For this setup, Rescaling is applied before the base model.
preprocess_input = Rescaling(1./255) # Or tf.keras.applications.mobilenet_v2.preprocess_input

# Apply augmentation and preprocessing
# Using AUTOTUNE for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE) # No augmentation for validation

# Configure dataset for performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. Define the Model (Transfer Learning with MobileNetV2) ---
print("Building the model...")

# Load pre-trained MobileNetV2 model, excluding the top classification layer
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                         include_top=False, # Do not include ImageNet classifier at the top
                         weights='imagenet')

# Freeze the base model layers (so we don't re-train them initially)
base_model.trainable = False

# Add our custom classification head
model = Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax') # Output layer with softmax for multi-class
])

# --- 4. Compile the Model ---
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy', # Use categorical_crossentropy for one-hot encoded labels
              metrics=['accuracy'])

model.summary()

# --- 5. Train the Model ---
print("Training the model...")

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[early_stopping, model_checkpoint]
)

# --- (Optional) Fine-tuning ---
# Unfreeze some layers of the base model and train with a very low learning rate
# This is often done after the initial training phase.
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100 # Example: unfreeze layers from block 100 onwards

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE / 10), # Lower LR
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Fine-tuning the model...")
history_fine_tune = model.fit(
    train_ds,
    epochs=EPOCHS + 10, # Train for a few more epochs, e.g., 10 more
    initial_epoch=history.epoch[-1] + 1, # Start from where previous training stopped
    validation_data=val_ds,
    callbacks=[early_stopping, model_checkpoint] # Can reuse or adjust callbacks
)


# --- 6. Evaluate the Model ---
print("Evaluating the model...")
# Load the best model saved by ModelCheckpoint
best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
loss, accuracy = best_model.evaluate(val_ds)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy*100:.2f}%")

# --- 7. Plot Training History ---
def plot_history(history, history_fine_tune=None):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    if history_fine_tune:
        acc += history_fine_tune.history['accuracy']
        val_acc += history_fine_tune.history['val_accuracy']
        loss += history_fine_tune.history['loss']
        val_loss += history_fine_tune.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_history(history, history_fine_tune)

# --- 8. Make Predictions on New Images ---
def predict_image(image_path, model_to_use, class_names_list):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # Preprocess the image (same as training)
    img_array = preprocess_input(img_array) # Apply Rescaling

    predictions = model_to_use.predict(img_array)
    score = tf.nn.softmax(predictions[0]) # Apply softmax if not already (model output is already softmax)
    predicted_class_index = np.argmax(score)
    predicted_class = class_names_list[predicted_class_index]
    confidence = 100 * np.max(score)

    print(
        f"This image most likely belongs to '{predicted_class}' with a {confidence:.2f}% confidence."
    )
    # plt.imshow(img) # Display the image
    # plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    # plt.axis("off")
    # plt.show()
    return predicted_class, confidence

# Example prediction (replace with an actual image path)
# Create a dummy image for prediction if you don't have one
if not os.path.exists("sample_test_image.jpg"):
    try:
        # Try to pick one from dummy data if it exists
        sample_image_dir = os.path.join(DATA_DIR, class_names[0])
        if os.path.exists(sample_image_dir) and len(os.listdir(sample_image_dir)) > 0:
            sample_image_path = os.path.join(sample_image_dir, os.listdir(sample_image_dir)[0])
            shutil.copy(sample_image_path, "sample_test_image.jpg")
            print(f"Copied '{sample_image_path}' to 'sample_test_image.jpg' for prediction example.")
        else:
            dummy_pred_img_array = np.random.randint(0, 256, size=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            pred_img = tf.keras.preprocessing.image.array_to_img(dummy_pred_img_array)
            pred_img.save("sample_test_image.jpg")
            print("Created 'sample_test_image.jpg' for prediction example.")
    except Exception as e:
        print(f"Could not create or copy a sample image for prediction: {e}")


if os.path.exists("sample_test_image.jpg"):
    print("\n--- Making a prediction on a sample image ---")
    # Make sure to use the best_model (loaded from checkpoint) for prediction
    predict_image("sample_test_image.jpg", best_model, class_names)
else:
    print("\nSkipping prediction as 'sample_test_image.jpg' was not found or created.")

print(f"\nModel saved to: {MODEL_SAVE_PATH}")
print("To use the model later:")
print("loaded_model = tf.keras.models.load_model('laptop_tablet_phone_classifier.keras')")
print("# Then use loaded_model.predict(...) as shown in predict_image function.")