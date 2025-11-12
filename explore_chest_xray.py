"""
Step 1: Explore Chest X-Ray Pneumonia Dataset
This script helps you understand your new dataset BEFORE modifying the CNN
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


print("=" * 60)
print(" CHEST X-RAY PNEUMONIA DATASET EXPLORATION")
print("=" * 60)

print("""
INSTRUCTIONS:
1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

2. Extract it to a folder called 'chest_xray'

3. The structure should look like:
   chest_xray/
   ├── train/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── test/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── val/
       ├── NORMAL/
       └── PNEUMONIA/

4. Run this script to explore the dataset!
""")

# Define dataset path
dataset_path = 'dataset/chest_xray'

# ==============================================================================
# TASK 1: CHECK IF DATASET EXISTS
# ==============================================================================

print("\n[TASK 1] Checking if dataset exists...")

if not os.path.exists(dataset_path):
    print(f"\n❌ Dataset not found at '{dataset_path}'")
    print("Please download and extract the dataset first!")
    print("\nOnce downloaded, this script will help you explore it.")
    exit()
else:
    print(f"✓ Found dataset at '{dataset_path}'")

# ==============================================================================
# TASK 2: EXPLORE DATASET STRUCTURE
# ==============================================================================

print("\n" + "=" * 60)
print("[TASK 2] Exploring Dataset Structure")
print("=" * 60)


def count_images(folder_path):
    """Count images in a folder"""
    if not os.path.exists(folder_path):
        return 0
    return len([f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])


# Count images in each split
train_normal = count_images(os.path.join(dataset_path, 'train', 'NORMAL'))
train_pneumonia = count_images(os.path.join(dataset_path, 'train', 'PNEUMONIA'))
test_normal = count_images(os.path.join(dataset_path, 'test', 'NORMAL'))
test_pneumonia = count_images(os.path.join(dataset_path, 'test', 'PNEUMONIA'))
val_normal = count_images(os.path.join(dataset_path, 'val', 'NORMAL'))
val_pneumonia = count_images(os.path.join(dataset_path, 'val', 'PNEUMONIA'))

print("\n📊 Dataset Statistics:")
print(f"\nTraining Set:")
print(f"  ├─ NORMAL:    {train_normal:,} images")
print(f"  ├─ PNEUMONIA: {train_pneumonia:,} images")
print(f"  └─ TOTAL:     {train_normal + train_pneumonia:,} images")

print(f"\nTest Set:")
print(f"  ├─ NORMAL:    {test_normal:,} images")
print(f"  ├─ PNEUMONIA: {test_pneumonia:,} images")
print(f"  └─ TOTAL:     {test_normal + test_pneumonia:,} images")

print(f"\nValidation Set:")
print(f"  ├─ NORMAL:    {val_normal:,} images")
print(f"  ├─ PNEUMONIA: {val_pneumonia:,} images")
print(f"  └─ TOTAL:     {val_normal + val_pneumonia:,} images")

# Calculate class balance
total_normal = train_normal + test_normal + val_normal
total_pneumonia = train_pneumonia + test_pneumonia + val_pneumonia
total = total_normal + total_pneumonia

print(f"\n📈 Overall Class Distribution:")
print(f"  NORMAL:    {total_normal:,} ({total_normal / total * 100:.1f}%)")
print(f"  PNEUMONIA: {total_pneumonia:,} ({total_pneumonia / total * 100:.1f}%)")

# ==============================================================================
# TASK 3: COMPARE WITH MNIST
# ==============================================================================

print("\n" + "=" * 60)
print("[TASK 3] Comparing with MNIST Dataset")
print("=" * 60)

print("""
Let's compare this with what you learned from MNIST:

MNIST Dataset:
  • Images: 60,000 training + 10,000 test
  • Classes: 10 (digits 0-9)
  • Image size: 28×28 pixels
  • Color: Grayscale (1 channel)
  • Balanced: ~6,000 images per class

Chest X-Ray Dataset:
  • Images: {} training + {} test
  • Classes: 2 (Normal vs Pneumonia)
  • Image size: ??? (we'll check next!)
  • Color: ??? 
  • Balanced: {}
""".format(
    train_normal + train_pneumonia,
    test_normal + test_pneumonia,
    "Yes" if abs(total_normal - total_pneumonia) < total * 0.1 else "No - IMBALANCED!"
))

# ==============================================================================
# TASK 4: EXAMINE ACTUAL IMAGES
# ==============================================================================

print("\n" + "=" * 60)
print("[TASK 4] Examining Image Properties")
print("=" * 60)

# Get sample images
train_normal_path = os.path.join(dataset_path, 'train', 'NORMAL')
train_pneumonia_path = os.path.join(dataset_path, 'train', 'PNEUMONIA')

normal_images = [f for f in os.listdir(train_normal_path) if f.endswith('.jpeg')][:5]
pneumonia_images = [f for f in os.listdir(train_pneumonia_path) if f.endswith('.jpeg')][:5]

# Analyze image properties
print("\nAnalyzing sample images...")

image_sizes = []
image_modes = []

for img_file in normal_images + pneumonia_images:
    if img_file in normal_images:
        img_path = os.path.join(train_normal_path, img_file)
    else:
        img_path = os.path.join(train_pneumonia_path, img_file)

    img = Image.open(img_path)
    image_sizes.append(img.size)
    image_modes.append(img.mode)

# Find most common size and mode
from collections import Counter

size_counts = Counter(image_sizes)
mode_counts = Counter(image_modes)

print(f"\n📐 Image Properties:")
print(f"  Most common size: {size_counts.most_common(1)[0][0]} (width × height)")
print(f"  Color mode: {mode_counts.most_common(1)[0][0]}")
print(f"     • L = Grayscale")
print(f"     • RGB = Color (3 channels)")

# Show size distribution
print(f"\n  Size distribution (sample of {len(image_sizes)} images):")
for size, count in size_counts.most_common():
    print(f"    {size}: {count} images")

# ==============================================================================
# TASK 5: VISUALIZE SAMPLE IMAGES
# ==============================================================================

print("\n" + "=" * 60)
print("[TASK 5] Visualizing Sample Images")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.suptitle('Sample Chest X-Ray Images', fontsize=16, fontweight='bold')

# Plot normal images
for i, img_file in enumerate(normal_images):
    img_path = os.path.join(train_normal_path, img_file)
    img = Image.open(img_path)

    ax = axes[0, i]
    ax.imshow(img, cmap='gray')
    ax.set_title(f'NORMAL\nSize: {img.size[0]}×{img.size[1]}', fontsize=10, color='green')
    ax.axis('off')

# Plot pneumonia images
for i, img_file in enumerate(pneumonia_images):
    img_path = os.path.join(train_pneumonia_path, img_file)
    img = Image.open(img_path)

    ax = axes[1, i]
    ax.imshow(img, cmap='gray')
    ax.set_title(f'PNEUMONIA\nSize: {img.size[0]}×{img.size[1]}', fontsize=10, color='red')
    ax.axis('off')

plt.tight_layout()
plt.savefig('chest_xray_exploration.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization: chest_xray_exploration.png")

# ==============================================================================
# TASK 6: KEY DIFFERENCES SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("[TASK 6] KEY DIFFERENCES FROM MNIST")
print("=" * 60)

# Calculate typical image size
avg_width = np.mean([size[0] for size in image_sizes])
avg_height = np.mean([size[1] for size in image_sizes])

print(f"""
Now that you've explored the dataset, let's identify what needs to change:

1. IMAGE SIZE:
   MNIST:     28 × 28 pixels
   X-Ray:     ~{int(avg_width)} × {int(avg_height)} pixels

   ❓ QUESTION: Can we use the same input_shape=(28, 28, 1)?
   💡 THINK: What happens if we resize {int(avg_width)}×{int(avg_height)} to 28×28?
             Will we lose important medical details?

2. NUMBER OF CLASSES:
   MNIST:     10 classes (digits 0-9)
   X-Ray:     2 classes (Normal vs Pneumonia)

   ❓ QUESTION: What needs to change in the output layer?
   💡 THINK: How many neurons in the final Dense layer?

3. DATASET SIZE:
   MNIST:     60,000 training images
   X-Ray:     {train_normal + train_pneumonia:,} training images

   ❓ QUESTION: With less data, what techniques should we use?
   💡 THINK: Data augmentation? Different dropout? Transfer learning?

4. CLASS BALANCE:
   MNIST:     Balanced (~6,000 per class)
   X-Ray:     Normal: {train_normal}, Pneumonia: {train_pneumonia}

   ❓ QUESTION: Is this balanced? If not, what problems could this cause?
   💡 THINK: Will the model just predict the majority class?

5. DATA LOADING:
   MNIST:     Comes pre-packaged in Keras
   X-Ray:     Images in folders

   ❓ QUESTION: How do we load images from folders?
   💡 THINK: Need to read file paths, load images, assign labels

6. GRAYSCALE vs RGB:
   MNIST:     Grayscale (1 channel)
   X-Ray:     {mode_counts.most_common(1)[0][0]} mode

   ❓ QUESTION: Does this affect our preprocessing?
""")

# ==============================================================================
# TASK 7: REFLECTION QUESTIONS
# ==============================================================================

print("\n" + "=" * 60)
print("[TASK 7] REFLECTION - ANSWER THESE QUESTIONS")
print("=" * 60)

print("""
Before we modify the code, think about these questions:

1. Should we keep images at their original size (~1000×1000) or resize them?
   • What are the trade-offs?
   • Larger = more detail but slower training
   • Smaller = faster but lose detail

2. Looking at the class imbalance:
   • If {} PNEUMONIA and {} NORMAL
   • What accuracy would we get if we always predict "PNEUMONIA"?
   • How should we handle this?

3. With only ~{} training images (vs 60,000 in MNIST):
   • What techniques prevent overfitting?
   • Should we use data augmentation?
   • Should we use dropout?

4. The output layer currently has 10 neurons for MNIST.
   • How many neurons for binary classification (2 classes)?
   • Should we use 'softmax' or 'sigmoid'?

NEXT STEP: Once you understand these differences, we'll modify the code together!
""".format(train_pneumonia, train_normal, train_normal + train_pneumonia))

print("\n" + "=" * 60)
print("Exploration complete! Review the questions above.")
print("=" * 60)