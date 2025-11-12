import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print(" MNIST DIGIT CLASSIFICATION WITH CNN")
print("="*60)
print(f"TensorFlow version: {tf.__version__}\n")

#==============================================================================
# STEP 1: LOAD AND EXPLORE THE DATA
#==============================================================================

print("\n" + "="*60)
print(" STEP 1: LOADING MNIST DATASET")
print("="*60)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"\n✓ Dataset loaded successfully!")
print(f"  Training images: {X_train.shape[0]:,} images of {X_train.shape[1]}×{X_train.shape[2]} pixels")
print(f"  Test images: {X_test.shape[0]:,} images")
print(f"  Number of classes: {len(np.unique(y_train))} (digits 0-9)")

# Visualize some random examples
print("\n📊 Visualizing sample images...")
fig, axes = plt.subplots(3, 6, figsize=(15, 7))
fig.suptitle('Sample MNIST Digits', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    # Pick a random image
    idx = np.random.randint(0, len(X_train))
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(f"Label: {y_train[idx]}", fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig('01_mnist_examples.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: 01_mnist_examples.png")

#==============================================================================
# STEP 2: PREPROCESS THE DATA
#==============================================================================

print("\n" + "="*60)
print(" STEP 2: PREPROCESSING DATA")
print("="*60)

# 2.1: Reshape - Add channel dimension
# CNNs expect input shape: (height, width, channels)
# For grayscale images, channels = 1

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"\n✓ Reshaped data:")
print(f"  Training shape: {X_train.shape}")
print(f"  Test shape: {X_test.shape}")

# 2.2: Normalize pixel values
# Original range: 0-255 (8-bit grayscale)
# Normalized range: 0-1
# This helps the neural network learn faster

print(f"\n  Original pixel range: [{X_train.min():.0f}, {X_train.max():.0f}]")
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
print(f"  Normalized pixel range: [{X_train.min():.2f}, {X_train.max():.2f}]")

# 2.3: One-hot encode labels
# Convert integer labels to binary vectors
# Example: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print(f"\n✓ One-hot encoded labels:")
print(f"  Original label: {y_train[0]}")
print(f"  One-hot vector: {y_train_cat[0]}")

#==============================================================================
# STEP 3: BUILD THE CNN ARCHITECTURE
#==============================================================================

print("\n" + "="*60)
print(" STEP 3: BUILDING CNN ARCHITECTURE")
print("="*60)

print("\n📐 CNN Architecture Overview:")
print("  Input (28×28×1) → Conv → Pool → Conv → Pool → Flatten → Dense → Output (10)")

model = keras.Sequential([
    # ========== CONVOLUTIONAL BLOCK 1 ==========
    # Conv2D: Applies filters to detect basic patterns (edges, curves)
    # - 32 filters of size 3×3
    # - 'relu' activation adds non-linearity
    # - Input shape: (28, 28, 1)
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                  input_shape=(28, 28, 1), name='conv1'),

    # MaxPooling: Reduces spatial dimensions by taking maximum value
    # - Pool size 2×2 reduces 26×26 → 13×13
    # - Helps make the model more robust to translations
    layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),

    # ========== CONVOLUTIONAL BLOCK 2 ==========
    # More filters to detect complex patterns
    # - 64 filters capture higher-level features
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),

    # ========== FULLY CONNECTED LAYERS ==========
    # Flatten: Convert 2D feature maps to 1D vector
    layers.Flatten(name='flatten'),

    # Dense: Fully connected layer with 128 neurons
    # - Learns combinations of features
    layers.Dense(128, activation='relu', name='dense1'),

    # Dropout: Randomly drops 50% of connections during training
    # - Prevents overfitting (memorizing training data)
    # - Forces network to learn robust features
    layers.Dropout(0.5, name='dropout'),

    # ========== OUTPUT LAYER ==========
    # Dense: 10 neurons (one for each digit 0-9)
    # - Softmax converts outputs to probabilities (sum = 1)
    layers.Dense(10, activation='softmax', name='output')
])

# Display detailed model architecture
print("\n" + "-"*60)
model.summary()
print("-"*60)

# Count total parameters
trainable_params = model.count_params()
print(f"\n✓ Total trainable parameters: {trainable_params:,}")

#==============================================================================
# STEP 4: COMPILE THE MODEL
#==============================================================================

print("\n" + "="*60)
print(" STEP 4: COMPILING MODEL")
print("="*60)

# Configure the model for training
model.compile(
    # Optimizer: Algorithm that updates weights
    # Adam: Adaptive learning rate, works well in most cases
    optimizer='adam',

    # Loss function: Measures how wrong the predictions are
    # categorical_crossentropy: For multi-class classification
    loss='categorical_crossentropy',

    # Metrics: What to track during training
    metrics=['accuracy']
)

print("\n✓ Model compiled successfully!")
print("  Optimizer: Adam")
print("  Loss: Categorical Crossentropy")
print("  Metrics: Accuracy")

#==============================================================================
# STEP 5: TRAIN THE MODEL
#==============================================================================

print("\n" + "="*60)
print(" STEP 5: TRAINING MODEL")
print("="*60)
print("\nThis may take a few minutes...\n")

# Train the model
history = model.fit(
    X_train, y_train_cat,          # Training data
    batch_size=128,                # Process 128 images at a time
    epochs=10,                     # Complete passes through dataset
    validation_split=0.1,          # Use 10% for validation
    verbose=1                      # Show progress bar
)

print("\n✓ Training completed!")

#==============================================================================
# STEP 6: EVALUATE THE MODEL
#==============================================================================

print("\n" + "="*60)
print(" STEP 6: EVALUATING MODEL ON TEST SET")
print("="*60)

# Evaluate on test set (unseen data)
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)

print(f"\n🎯 Final Results:")
print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  Test Loss: {test_loss:.4f}")

# Performance interpretation
if test_accuracy > 0.99:
    print("\n  🌟 Excellent! Your model is performing very well!")
elif test_accuracy > 0.95:
    print("\n  ✓ Good! The model has learned to classify digits effectively.")
else:
    print("\n  ⚠ Moderate performance. Consider training longer or adjusting architecture.")

# ==============================================================================
# STEP 7: VISUALIZE TRAINING HISTORY
# ==============================================================================

print("\n" + "=" * 60)
print(" STEP 7: VISUALIZING TRAINING PROGRESS")
print("=" * 60)

# Create comprehensive training visualization
fig = plt.figure(figsize=(16, 6))

# Plot 1: Accuracy over epochs
ax1 = plt.subplot(1, 3, 1)
ax1.plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2, marker='o')
ax1.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2, marker='s')
ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.9, 1.0])

# Plot 2: Loss over epochs
ax2 = plt.subplot(1, 3, 2)
ax2.plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2, marker='o')
ax2.plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s')
ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Final metrics bar chart
ax3 = plt.subplot(1, 3, 3)
metrics = ['Training', 'Validation', 'Test']
accuracies = [history.history['accuracy'][-1],
              history.history['val_accuracy'][-1],
              test_accuracy]
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax3.bar(metrics, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=12)
ax3.set_ylim([0.9, 1.0])
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2., height,
             f'{acc * 100:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('02_training_history.png', dpi=150, bbox_inches='tight')
print("\n  ✓ Saved: 02_training_history.png")

# ==============================================================================
# STEP 8: TEST PREDICTIONS AND VISUALIZE RESULTS
# ==============================================================================

print("\n" + "=" * 60)
print(" STEP 8: TESTING PREDICTIONS")
print("=" * 60)

# Make predictions on test set
predictions = model.predict(X_test[:20], verbose=0)

# Visualize predictions with confidence
fig = plt.figure(figsize=(16, 8))
fig.suptitle('Model Predictions on Test Images', fontsize=16, fontweight='bold')

for i in range(20):
    ax = plt.subplot(4, 5, i + 1)

    # Display image
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')

    # Get prediction and confidence
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]
    confidence = predictions[i][predicted_label] * 100

    # Color code: green for correct, red for incorrect
    if predicted_label == true_label:
        color = 'green'
        symbol = '✓'
    else:
        color = 'red'
        symbol = '✗'

    # Title with prediction info
    ax.set_title(f"{symbol} Pred: {predicted_label} ({confidence:.1f}%)\nTrue: {true_label}",
                 color=color, fontsize=10, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('03_predictions.png', dpi=150, bbox_inches='tight')
print("\n  ✓ Saved: 03_predictions.png")

# Calculate and display confusion examples
predictions_all = model.predict(X_test, verbose=0)
predicted_labels = np.argmax(predictions_all, axis=1)
mistakes = np.where(predicted_labels != y_test)[0]

print(f"\n📊 Prediction Statistics:")
print(f"  Correct predictions: {len(y_test) - len(mistakes)}/{len(y_test)}")
print(f"  Incorrect predictions: {len(mistakes)}/{len(y_test)}")
print(f"  Error rate: {len(mistakes) / len(y_test) * 100:.2f}%")

# ==============================================================================
# STEP 9: ANALYZE MISCLASSIFICATIONS
# ==============================================================================

if len(mistakes) > 0:
    print("\n" + "=" * 60)
    print(" STEP 9: ANALYZING MISCLASSIFICATIONS")
    print("=" * 60)

    # Show some mistakes
    num_mistakes_to_show = min(10, len(mistakes))
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle('Examples of Misclassified Digits', fontsize=16, fontweight='bold')

    for i in range(num_mistakes_to_show):
        idx = mistakes[i]
        ax = plt.subplot(2, 5, i + 1)

        ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')

        pred = predicted_labels[idx]
        true = y_test[idx]
        conf = predictions_all[idx][pred] * 100

        ax.set_title(f"Predicted: {pred} ({conf:.1f}%)\nActual: {true}",
                     color='red', fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('04_misclassifications.png', dpi=150, bbox_inches='tight')
    print("\n  ✓ Saved: 04_misclassifications.png")

# ==============================================================================
# STEP 10: SAVE THE MODEL
# ==============================================================================

print("\n" + "=" * 60)
print(" STEP 10: SAVING TRAINED MODEL")
print("=" * 60)

# Save the entire model
model.save('mnist_cnn_model.keras')
print("\n  ✓ Saved: mnist_cnn_model.keras")

# Also save in older format for compatibility
model.save('mnist_cnn_model.h5')
print("  ✓ Saved: mnist_cnn_model.h5")

print("\n💾 Model saved successfully! You can load it later with:")
print("  model = keras.models.load_model('mnist_cnn_model.keras')")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print(" 🎉 TRAINING COMPLETE! ")
print("=" * 60)

print(f"\n📈 Final Performance Summary:")
print(f"  ├─ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"  ├─ Total Parameters: {trainable_params:,}")
print(f"  └─ Training Time: {len(history.history['loss'])} epochs")

print(f"\n📁 Generated Files:")
print(f"  ├─ 01_mnist_examples.png (sample training images)")
print(f"  ├─ 02_training_history.png (training curves)")
print(f"  ├─ 03_predictions.png (model predictions)")
if len(mistakes) > 0:
    print(f"  ├─ 04_misclassifications.png (error analysis)")
print(f"  ├─ mnist_cnn_model.keras (trained model)")
print(f"  └─ mnist_cnn_model.h5 (trained model - compatible)")

print("\n" + "=" * 60)
print(" NEXT STEPS:")
print("=" * 60)
print("""
1. Try testing on your own handwritten digits
2. Experiment with different architectures
3. Adjust hyperparameters (epochs, batch size, dropout)
4. Try transfer learning on other datasets
""")

print("✨ Great job on completing your first CNN project! ✨\n")