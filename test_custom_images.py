"""
Test Trained MNIST Model on Custom Images
This script loads your trained model and tests it on new handwritten digits
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os

print("=" * 60)
print(" TEST CUSTOM HANDWRITTEN DIGITS")
print("=" * 60)

# ==============================================================================
# STEP 1: LOAD THE TRAINED MODEL
# ==============================================================================

print("\n[1] Loading trained model...")

try:
    model = keras.models.load_model('mnist_cnn_model.keras')
    print("  ✓ Model loaded successfully!")
except:
    try:
        model = keras.models.load_model('mnist_cnn_model.h5')
        print("  ✓ Model loaded successfully (from .h5 file)!")
    except:
        print("  ✗ Error: Could not find trained model!")
        print("  Please run 'mnist_cnn_complete_tutorial.py' first to train a model.")
        exit()

model.summary()


# ==============================================================================
# STEP 2: PREPROCESS CUSTOM IMAGES
# ==============================================================================

def preprocess_custom_image(image_path):
    """
    Preprocess a custom image to match MNIST format:
    - Convert to grayscale
    - Resize to 28×28 pixels
    - Normalize pixel values to 0-1
    - Invert colors if needed (MNIST has white digits on black background)
    """
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale

    # Resize to 28×28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img)

    # Check if we need to invert (MNIST is white digit on black background)
    # If the average pixel value is > 127, the image is likely light background
    if np.mean(img_array) > 127:
        img_array = 255 - img_array  # Invert

    # Normalize to 0-1
    img_array = img_array.astype('float32') / 255.0

    # Reshape to (1, 28, 28, 1) for model input
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


# ==============================================================================
# STEP 3: TEST ON CUSTOM IMAGES
# ==============================================================================

def predict_digit(model, image_path):
    """
    Predict the digit in a custom image
    """
    # Preprocess the image
    img_array = preprocess_custom_image(image_path)

    # Make prediction
    prediction = model.predict(img_array, verbose=0)

    # Get predicted digit and confidence
    predicted_digit = np.argmax(prediction[0])
    confidence = prediction[0][predicted_digit] * 100

    return predicted_digit, confidence, prediction[0], img_array


# ==============================================================================
# EXAMPLE: Test on multiple custom images
# ==============================================================================

print("\n[2] Testing on custom images...")
print("\nTo test your own images:")
print("  1. Create a folder called 'custom_digits'")
print("  2. Put your handwritten digit images in it")
print("  3. Make sure images have clear digits (any format: jpg, png, etc.)")
print("\n" + "-" * 60)

# Check if custom images folder exists
if os.path.exists('custom_digits'):
    image_files = [f for f in os.listdir('custom_digits')
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if len(image_files) > 0:
        print(f"\nFound {len(image_files)} custom images!")

        # Create visualization
        num_images = min(12, len(image_files))
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle('Predictions on Your Custom Images', fontsize=16, fontweight='bold')

        for i, img_file in enumerate(image_files[:num_images]):
            img_path = os.path.join('custom_digits', img_file)

            try:
                # Predict
                pred_digit, confidence, probs, processed_img = predict_digit(model, img_path)

                # Plot
                ax = plt.subplot(3, 4, i + 1)
                ax.imshow(processed_img.reshape(28, 28), cmap='gray')
                ax.set_title(f"Predicted: {pred_digit}\nConfidence: {confidence:.1f}%",
                             fontsize=11, fontweight='bold')
                ax.axis('off')

                print(f"\n  Image: {img_file}")
                print(f"    Predicted: {pred_digit} (confidence: {confidence:.1f}%)")
                print(f"    All probabilities: {[f'{p * 100:.1f}%' for p in probs]}")

            except Exception as e:
                print(f"\n  ✗ Error processing {img_file}: {e}")

        plt.tight_layout()
        plt.savefig('custom_predictions.png', dpi=150, bbox_inches='tight')
        print("\n  ✓ Saved visualization: custom_predictions.png")
    else:
        print("\n  No image files found in 'custom_digits' folder")
else:
    print("\n  Folder 'custom_digits' not found.")
    print("  Creating example demonstration instead...")

# ==============================================================================
# DEMO: Create and test synthetic digits
# ==============================================================================

print("\n[3] Creating demo synthetic digits for testing...")

# Create some simple synthetic digits for demo
from tensorflow.keras.datasets import mnist

(_, _), (X_test_mnist, y_test_mnist) = mnist.load_data()

# Take some random test images
demo_indices = np.random.choice(len(X_test_mnist), 12, replace=False)

fig = plt.figure(figsize=(16, 8))
fig.suptitle('Demo: Predictions on MNIST Test Images', fontsize=16, fontweight='bold')

for i, idx in enumerate(demo_indices):
    # Prepare image
    img = X_test_mnist[idx].reshape(1, 28, 28, 1).astype('float32') / 255.0
    true_label = y_test_mnist[idx]

    # Predict
    prediction = model.predict(img, verbose=0)
    predicted_digit = np.argmax(prediction[0])
    confidence = prediction[0][predicted_digit] * 100

    # Plot
    ax = plt.subplot(3, 4, i + 1)
    ax.imshow(img.reshape(28, 28), cmap='gray')

    color = 'green' if predicted_digit == true_label else 'red'
    symbol = '✓' if predicted_digit == true_label else '✗'

    ax.set_title(f"{symbol} Pred: {predicted_digit} ({confidence:.1f}%)\nTrue: {true_label}",
                 color=color, fontsize=11, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('demo_predictions.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: demo_predictions.png")


# ==============================================================================
# INTERACTIVE TESTING FUNCTION
# ==============================================================================

def test_single_image_interactive(image_path):
    """
    Test a single image and show detailed results
    """
    print(f"\n{'=' * 60}")
    print(f" Testing: {image_path}")
    print('=' * 60)

    # Predict
    pred_digit, confidence, probs, processed_img = predict_digit(model, image_path)

    # Display results
    print(f"\n🎯 Prediction: {pred_digit}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print(f"\nProbability distribution:")
    for digit in range(10):
        bar = '█' * int(probs[digit] * 50)
        print(f"  {digit}: {bar} {probs[digit] * 100:5.2f}%")

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Show processed image
    ax1.imshow(processed_img.reshape(28, 28), cmap='gray')
    ax1.set_title(f'Processed Image\n(28×28 grayscale)', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Show probability distribution
    ax2.bar(range(10), probs, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Digit', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(10))
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'prediction_{os.path.basename(image_path)}.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: prediction_{os.path.basename(image_path)}.png")


# ==============================================================================
# FINAL INSTRUCTIONS
# ==============================================================================

print("\n" + "=" * 60)
print(" HOW TO USE THIS SCRIPT")
print("=" * 60)

print("""
METHOD 1: Test your own images
  1. Create folder: mkdir custom_digits
  2. Add your digit images to the folder
  3. Run this script: python test_custom_images.py

METHOD 2: Test a single image in code
  Add this code:
    test_single_image_interactive('path/to/your/image.png')

METHOD 3: Use the function in your code
  from test_custom_images import predict_digit
  digit, confidence, probs, img = predict_digit(model, 'image.png')
  print(f"Predicted digit: {digit} (confidence: {confidence:.1f}%)")

TIPS FOR BEST RESULTS:
  • Use clear, well-lit images
  • Digit should be centered
  • Black or white background works best
  • Image can be any size (will be resized to 28×28)
  • Model works best on handwritten digits similar to MNIST style
""")

print("\n✨ Ready to classify your handwritten digits! ✨\n")