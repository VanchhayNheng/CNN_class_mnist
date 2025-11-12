"""
Chest X-Ray Pneumonia Classification with CNN
Modified from MNIST classifier - now you understand WHY each change was made!

YOUR DESIGN DECISIONS:
✓ Image size: 224×224 (preserves patterns, proven in research)
✓ Output neurons: 2 (one per class)
✓ Class weights: Handle imbalance (1341 Normal vs 3875 Pneumonia)
✓ Early stopping: patience=5 to prevent overfitting
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print(" CHEST X-RAY PNEUMONIA CLASSIFICATION WITH CNN")
print("=" * 70)
print(f"TensorFlow version: {tf.__version__}\n")

# ==============================================================================
# GPU CONFIGURATION - ADD THIS FIRST!
# ==============================================================================

print("=" * 70)
print(" GPU SETUP")
print("=" * 70)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs detected: {len(gpus)}")

if len(gpus) > 0:
    print("✓ GPU available!")
    for gpu in gpus:
        print(f"  {gpu}")

    # Enable memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ Memory growth enabled")
    except RuntimeError as e:
        print(f"⚠ {e}")

    # Use mixed precision for faster training on modern GPUs
    from tensorflow.keras import mixed_precision

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("✓ Mixed precision enabled (faster on modern GPUs)")

    USE_GPU = True
    BATCH_SIZE = 64  # Larger batch for GPU
    print("\n🚀 Training will use GPU!")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Estimated time: ~5-10 minutes")
else:
    print("⚠ No GPU - using CPU")
    USE_GPU = False
    BATCH_SIZE = 32  # Smaller batch for CPU
    print(f"\n   Batch size: {BATCH_SIZE}")
    print(f"   Estimated time: ~30-60 minutes")

print("=" * 70 + "\n")

# ==============================================================================
# MAIN TRAINING CODE
# ==============================================================================

print("=" * 70)
print(" CHEST X-RAY PNEUMONIA CLASSIFICATION")
print("=" * 70)
print(f"TensorFlow version: {tf.__version__}")
print(f"Training device: {'GPU' if USE_GPU else 'CPU'}\n")

# [REST OF THE CODE IS THE SAME AS chest_xray_classifier.py]
# Just with BATCH_SIZE variable instead of hardcoded 32


# ==============================================================================
# STEP 1: LOAD DATA FROM FOLDERS
# CHANGE FROM MNIST: Now we load from folder structure instead of keras dataset
# ==============================================================================

print("\n" + "=" * 70)
print(" STEP 1: LOADING DATA FROM FOLDERS")
print("=" * 70)

dataset_path = 'dataset/chest_xray'

if not os.path.exists(dataset_path):
    print("\n❌ Dataset not found!")
    print("Please download the dataset from:")
    print("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("\nExtract it to a folder called 'chest_xray' in the same directory.")
    exit()


def load_images_from_folder(folder_path, label, target_size=(224, 224)):
    """
    Load images from a folder and assign them a label

    WHY THIS IS DIFFERENT FROM MNIST:
    - MNIST: Pre-packaged arrays
    - Chest X-Ray: Individual image files in folders
    """
    images = []
    labels = []

    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpeg')]

    print(f"  Loading {len(image_files)} images from {os.path.basename(folder_path)}...")

    for img_file in image_files:
        try:
            # Load image
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path).convert('L')  # Convert to grayscale

            # Resize to 224×224
            # YOU DECIDED: 224×224 preserves patterns while being computationally feasible
            img = img.resize(target_size, Image.Resampling.LANCZOS)

            # Convert to numpy array
            img_array = np.array(img)

            images.append(img_array)
            labels.append(label)

        except Exception as e:
            print(f"    ⚠ Error loading {img_file}: {e}")

    return np.array(images), np.array(labels)


# Load training data
print("\n📁 Loading TRAINING data...")
train_normal_path = os.path.join(dataset_path, 'train', 'NORMAL')
train_pneumonia_path = os.path.join(dataset_path, 'train', 'PNEUMONIA')

X_train_normal, y_train_normal = load_images_from_folder(train_normal_path, label=0)
X_train_pneumonia, y_train_pneumonia = load_images_from_folder(train_pneumonia_path, label=1)

# Combine
X_train = np.concatenate([X_train_normal, X_train_pneumonia], axis=0)
y_train = np.concatenate([y_train_normal, y_train_pneumonia], axis=0)

print(f"✓ Training data loaded: {X_train.shape}")

# Load test data
print("\n📁 Loading TEST data...")
test_normal_path = os.path.join(dataset_path, 'test', 'NORMAL')
test_pneumonia_path = os.path.join(dataset_path, 'test', 'PNEUMONIA')

X_test_normal, y_test_normal = load_images_from_folder(test_normal_path, label=0)
X_test_pneumonia, y_test_pneumonia = load_images_from_folder(test_pneumonia_path, label=1)

# Combine
X_test = np.concatenate([X_test_normal, X_test_pneumonia], axis=0)
y_test = np.concatenate([y_test_normal, y_test_pneumonia], axis=0)

print(f"✓ Test data loaded: {X_test.shape}")

# Shuffle data
print("\n🔀 Shuffling data...")
train_indices = np.random.permutation(len(X_train))
X_train = X_train[train_indices]
y_train = y_train[train_indices]

test_indices = np.random.permutation(len(X_test))
X_test = X_test[test_indices]
y_test = y_test[test_indices]

print(f"\n✓ Dataset Summary:")
print(f"  Training: {len(X_train)} images")
print(f"    - NORMAL: {np.sum(y_train == 0)}")
print(f"    - PNEUMONIA: {np.sum(y_train == 1)}")
print(f"  Test: {len(X_test)} images")
print(f"    - NORMAL: {np.sum(y_test == 0)}")
print(f"    - PNEUMONIA: {np.sum(y_test == 1)}")

# ==============================================================================
# STEP 2: PREPROCESS DATA
# SIMILAR TO MNIST: Normalize and reshape, but different dimensions!
# ==============================================================================

print("\n" + "=" * 70)
print(" STEP 2: PREPROCESSING DATA")
print("=" * 70)

# Reshape to add channel dimension
# CHANGE FROM MNIST: (28, 28, 1) → (224, 224, 1)
print(f"\n📐 Original shape: {X_train.shape}")
X_train = X_train.reshape(-1, 224, 224, 1)
X_test = X_test.reshape(-1, 224, 224, 1)
print(f"  After reshape: {X_train.shape}")

# Normalize pixel values to [0, 1]
# SAME AS MNIST: This step doesn't change!
print(f"\n  Original pixel range: [{X_train.min()}, {X_train.max()}]")
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
print(f"  Normalized range: [{X_train.min():.2f}, {X_train.max():.2f}]")

# One-hot encode labels
# CHANGE FROM MNIST: 10 classes → 2 classes
# YOU UNDERSTOOD: Each neuron represents one class!
y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)

print(f"\n✓ One-hot encoding:")
print(f"  Original label: {y_train[0]} (0=Normal, 1=Pneumonia)")
print(f"  One-hot vector: {y_train_cat[0]}")

# ==============================================================================
# STEP 3: CALCULATE CLASS WEIGHTS
# NEW FOR MEDICAL IMAGING: Handle class imbalance!
# ==============================================================================

print("\n" + "=" * 70)
print(" STEP 3: CALCULATING CLASS WEIGHTS")
print("=" * 70)

# Count samples per class
n_normal = np.sum(y_train == 0)
n_pneumonia = np.sum(y_train == 1)
total_samples = len(y_train)

# Calculate weights (inverse frequency)
# YOU CALCULATED: Normal gets higher weight (minority class)
weight_normal = total_samples / (2 * n_normal)
weight_pneumonia = total_samples / (2 * n_pneumonia)

class_weight = {
    0: weight_normal,  # NORMAL (minority)
    1: weight_pneumonia  # PNEUMONIA (majority)
}

print(f"\n📊 Class Distribution:")
print(f"  NORMAL:    {n_normal} images ({n_normal / total_samples * 100:.1f}%)")
print(f"  PNEUMONIA: {n_pneumonia} images ({n_pneumonia / total_samples * 100:.1f}%)")

print(f"\n⚖️ Class Weights (to balance learning):")
print(f"  NORMAL weight:    {weight_normal:.3f} (higher = more important)")
print(f"  PNEUMONIA weight: {weight_pneumonia:.3f}")

print(f"\n💡 Without weights, model could get {max(n_normal, n_pneumonia) / total_samples * 100:.1f}%")
print(f"   accuracy by always predicting the majority class!")

# ==============================================================================
# STEP 4: BUILD CNN ARCHITECTURE
# MODIFIED FROM MNIST: Larger input, fewer output neurons
# ==============================================================================

print("\n" + "=" * 70)
print(" STEP 4: BUILDING CNN ARCHITECTURE")
print("=" * 70)

print("\n🏗️ Architecture Overview:")
print("  Input (224×224×1) → Conv → Pool → Conv → Pool → Flatten → Dense → Output (2)")
print("\n  KEY CHANGES FROM MNIST:")
print("    • Input: (28,28,1) → (224,224,1)")
print("    • Output: 10 neurons → 2 neurons")

model = keras.Sequential([
    # ========== CONVOLUTIONAL BLOCK 1 ==========
    # CHANGE: input_shape now (224, 224, 1) instead of (28, 28, 1)
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                  input_shape=(224, 224, 1), name='conv1'),
    layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
    # After pool1: 224×224 → 112×112

    # ========== CONVOLUTIONAL BLOCK 2 ==========
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
    # After pool2: 112×112 → 56×56

    # ========== CONVOLUTIONAL BLOCK 3 (NEW!) ==========
    # For larger images (224×224), we can add more layers
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv3'),
    layers.MaxPooling2D(pool_size=(2, 2), name='pool3'),
    # After pool3: 56×56 → 28×28

    # ========== FLATTEN ==========
    layers.Flatten(name='flatten'),

    # ========== FULLY CONNECTED LAYERS ==========
    layers.Dense(128, activation='relu', name='dense1'),
    layers.Dropout(0.5, name='dropout'),

    # ========== OUTPUT LAYER ==========
    # CHANGE: 10 neurons → 2 neurons (YOU UNDERSTOOD WHY!)
    # 2 neurons for 2 classes: [P(Normal), P(Pneumonia)]
    layers.Dense(2, activation='softmax', name='output')
])

# Display architecture
print("\n" + "-" * 70)
model.summary()
print("-" * 70)

trainable_params = model.count_params()
print(f"\n✓ Total parameters: {trainable_params:,}")
print(f"  (More than MNIST because of larger input images!)")

# ==============================================================================
# STEP 5: COMPILE MODEL
# SAME AS MNIST: Configuration works for binary classification too!
# ==============================================================================

print("\n" + "=" * 70)
print(" STEP 5: COMPILING MODEL")
print("=" * 70)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Works for 2+ classes
    metrics=['accuracy']
)

print("\n✓ Model compiled!")
print("  Optimizer: Adam")
print("  Loss: Categorical Crossentropy")
print("  Metrics: Accuracy")

# ==============================================================================
# STEP 6: SETUP CALLBACKS
# NEW: Early stopping, model checkpoint, learning rate reduction
# ==============================================================================

print("\n" + "=" * 70)
print(" STEP 6: SETTING UP CALLBACKS")
print("=" * 70)

callbacks = [
    # Early stopping - YOU DECIDED: patience=5
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    # Save best model
    ModelCheckpoint(
        'best_chest_xray_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),

    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-7
    )
]

print("\n✓ Callbacks configured:")
print("  • Early Stopping (patience=5)")
print("  • Model Checkpoint (save best)")
print("  • Learning Rate Reduction")

# ==============================================================================
# STEP 7: TRAIN THE MODEL
# NEW: Using class_weight and callbacks!
# ==============================================================================

print("\n" + "=" * 70)
print(" STEP 7: TRAINING MODEL")
print("=" * 70)
print("\nThis will take ~30 minutes on CPU...")
print("Watch for:")
print("  • Training accuracy increasing")
print("  • Validation loss decreasing")
print("  • Early stopping if overfitting\n")

history = model.fit(
    X_train, y_train_cat,
    batch_size=32,  # Smaller batch for larger images
    epochs=50,  # More epochs, but early stopping will kick in
    validation_split=0.2,  # 20% for validation
    class_weight=class_weight,  # Handle imbalance!
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training completed!")

# ==============================================================================
# STEP 8: EVALUATE ON TEST SET
# ==============================================================================

print("\n" + "=" * 70)
print(" STEP 8: EVALUATING ON TEST SET")
print("=" * 70)

# Load best model
model = keras.models.load_model('best_chest_xray_model.keras')

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)

print(f"\n🎯 Test Results:")
print(f"  Accuracy: {test_accuracy * 100:.2f}%")
print(f"  Loss: {test_loss:.4f}")

# Get predictions
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification report
print("\n" + "-" * 70)
print(" DETAILED CLASSIFICATION REPORT")
print("-" * 70)
print("\n", classification_report(y_test, y_pred,
                                  target_names=['NORMAL', 'PNEUMONIA']))

# ==============================================================================
# STEP 9: VISUALIZE RESULTS
# ==============================================================================

print("\n" + "=" * 70)
print(" STEP 9: CREATING VISUALIZATIONS")
print("=" * 70)

# Plot 1: Training History
fig = plt.figure(figsize=(16, 6))

ax1 = plt.subplot(1, 3, 1)
ax1.plot(history.history['accuracy'], 'b-', label='Training', linewidth=2, marker='o')
ax1.plot(history.history['val_accuracy'], 'r-', label='Validation', linewidth=2, marker='s')
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(1, 3, 2)
ax2.plot(history.history['loss'], 'b-', label='Training', linewidth=2, marker='o')
ax2.plot(history.history['val_loss'], 'r-', label='Validation', linewidth=2, marker='s')
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Confusion Matrix
ax3 = plt.subplot(1, 3, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'])
ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('chest_xray_training_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: chest_xray_training_results.png")

# Plot 2: Sample Predictions
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
fig.suptitle('Sample Predictions on Test Set', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < len(X_test):
        ax.imshow(X_test[i].reshape(224, 224), cmap='gray')

        true_label = 'NORMAL' if y_test[i] == 0 else 'PNEUMONIA'
        pred_label = 'NORMAL' if y_pred[i] == 0 else 'PNEUMONIA'
        confidence = y_pred_probs[i][y_pred[i]] * 100

        color = 'green' if y_test[i] == y_pred[i] else 'red'
        symbol = '✓' if y_test[i] == y_pred[i] else '✗'

        ax.set_title(f"{symbol} Pred: {pred_label} ({confidence:.0f}%)\nTrue: {true_label}",
                     fontsize=9, color=color, fontweight='bold')
        ax.axis('off')

plt.tight_layout()
plt.savefig('chest_xray_predictions.png', dpi=150, bbox_inches='tight')
print("✓ Saved: chest_xray_predictions.png")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print(" 🎉 TRAINING COMPLETE!")
print("=" * 70)

print(f"\n📊 Final Performance:")
print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"  Total Parameters: {trainable_params:,}")
print(f"  Training Epochs: {len(history.history['loss'])}")

print(f"\n📁 Generated Files:")
print(f"  • best_chest_xray_model.keras (trained model)")
print(f"  • chest_xray_training_results.png (training curves + confusion matrix)")
print(f"  • chest_xray_predictions.png (sample predictions)")

print("\n" + "=" * 70)
print(" WHAT YOU LEARNED:")
print("=" * 70)
print("""
✓ How to adapt MNIST code to medical images
✓ Loading data from folder structures
✓ Handling class imbalance with weights
✓ Resizing images (224×224) while preserving patterns
✓ Understanding pixel redundancy vs information
✓ Output neurons = number of classes
✓ Using callbacks (early stopping, checkpoints)
✓ Evaluating medical classification models

NEXT STEPS:
1. Analyze the confusion matrix - which class is harder to predict?
2. Try data augmentation to improve accuracy
3. Experiment with different architectures
4. Test on individual X-ray images
""")

print("\n🌟 Excellent work! You've successfully built a medical image classifier! 🌟\n")