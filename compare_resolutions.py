"""
Visual Resolution Comparison for Chest X-Rays
This script helps you SEE what happens at different image sizes
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

print("=" * 70)
print(" VISUAL RESOLUTION COMPARISON - CHEST X-RAY IMAGES")
print("=" * 70)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

dataset_path = 'dataset/chest_xray'

# Different sizes to test
sizes_to_test = [
    (64, 64),
    (128, 128),
    (224, 224),
    (512, 512)
]

# ==============================================================================
# CHECK DATASET
# ==============================================================================

if not os.path.exists(dataset_path):
    print("\n❌ Dataset not found!")
    print("Please download the chest X-ray dataset first.")
    print("Run: python explore_chest_xray.py")
    exit()



def get_training_time_estimate(size):
    """Rough estimate of training time on CPU"""
    if size == (64, 64):
        return "~5 minutes (10 epochs)"
    elif size == (128, 128):
        return "~15 minutes (10 epochs)"
    elif size == (224, 224):
        return "~30 minutes (10 epochs)"
    elif size == (512, 512):
        return "~2 hours (10 epochs)"
    return "unknown"


print("=" * 70)
print("\n✅ Visual comparison complete!")
print("\nNext steps:")
print("  1. Open and study the 3 generated images")
print("  2. Answer the questions above")
print("  3. Tell me your decision!")
print("=" * 70)


# ==============================================================================
# LOAD SAMPLE IMAGES
# ==============================================================================

print("\n[1] Loading sample X-ray images...")

normal_path = os.path.join(dataset_path, 'train', 'NORMAL')
pneumonia_path = os.path.join(dataset_path, 'train', 'PNEUMONIA')

# Get one normal and one pneumonia image
normal_images = [f for f in os.listdir(normal_path) if f.endswith('.jpeg')]
pneumonia_images = [f for f in os.listdir(pneumonia_path) if f.endswith('.jpeg')]

# Load images
normal_img_path = os.path.join(normal_path, normal_images[0])
pneumonia_img_path = os.path.join(pneumonia_path, pneumonia_images[0])

normal_img = Image.open(normal_img_path).convert('L')
pneumonia_img = Image.open(pneumonia_img_path).convert('L')

print(f"✓ Loaded NORMAL image: {normal_img.size}")
print(f"✓ Loaded PNEUMONIA image: {pneumonia_img.size}")

# ==============================================================================
# CREATE COMPARISON VISUALIZATION
# ==============================================================================

print("\n[2] Creating resolution comparison...")

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 5, figure=fig, hspace=0.4, wspace=0.3)

# Title
fig.suptitle('Resolution Comparison: Can You Still See Important Patterns?',
             fontsize=18, fontweight='bold', y=0.98)

# Row 1: Original Images
ax_orig_normal = fig.add_subplot(gs[0, 0:2])
ax_orig_pneumonia = fig.add_subplot(gs[0, 3:5])

ax_orig_normal.imshow(normal_img, cmap='gray')
ax_orig_normal.set_title(
    f'ORIGINAL - NORMAL\nSize: {normal_img.size[0]}×{normal_img.size[1]} pixels\n(~3.9 million pixels)',
    fontsize=12, fontweight='bold', color='green')
ax_orig_normal.axis('off')

ax_orig_pneumonia.imshow(pneumonia_img, cmap='gray')
ax_orig_pneumonia.set_title(
    f'ORIGINAL - PNEUMONIA\nSize: {pneumonia_img.size[0]}×{pneumonia_img.size[1]} pixels\n(~3.9 million pixels)',
    fontsize=12, fontweight='bold', color='red')
ax_orig_pneumonia.axis('off')

# Row 2 & 3: Resized versions
row_positions = {
    (64, 64): (1, 0),
    (128, 128): (1, 1),
    (224, 224): (1, 2),
    (512, 512): (1, 3)
}

for size in sizes_to_test:
    # Resize images
    normal_resized = normal_img.resize(size, Image.Resampling.LANCZOS)
    pneumonia_resized = pneumonia_img.resize(size, Image.Resampling.LANCZOS)

    # Calculate percentage of original pixels
    original_pixels = normal_img.size[0] * normal_img.size[1]
    resized_pixels = size[0] * size[1]
    percentage = (resized_pixels / original_pixels) * 100

    # Normal image
    ax_normal = fig.add_subplot(gs[1, row_positions[size][1]])
    ax_normal.imshow(normal_resized, cmap='gray')
    ax_normal.set_title(f'{size[0]}×{size[1]}\n{resized_pixels:,} pixels\n({percentage:.1f}% of original)',
                        fontsize=10, fontweight='bold', color='green')
    ax_normal.axis('off')

    # Pneumonia image
    ax_pneumonia = fig.add_subplot(gs[2, row_positions[size][1]])
    ax_pneumonia.imshow(pneumonia_resized, cmap='gray')
    ax_pneumonia.set_title(f'{size[0]}×{size[1]}\n{resized_pixels:,} pixels\n({percentage:.1f}% of original)',
                           fontsize=10, fontweight='bold', color='red')
    ax_pneumonia.axis('off')

plt.savefig('resolution_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: resolution_comparison.png")

# ==============================================================================
# DETAILED COMPARISON - ZOOM INTO LUNG REGION
# ==============================================================================

print("\n[3] Creating detailed lung region comparison...")

# Let's zoom into a specific region to see detail better
fig2, axes = plt.subplots(2, 4, figsize=(18, 10))
fig2.suptitle('ZOOMED COMPARISON: Lung Region Detail', fontsize=16, fontweight='bold')

# Define crop region (center area of lungs)
crop_box_original = (
    normal_img.size[0] // 4,  # left
    normal_img.size[1] // 4,  # top
    3 * normal_img.size[0] // 4,  # right
    3 * normal_img.size[1] // 4  # bottom
)

for idx, size in enumerate(sizes_to_test):
    # Normal - resized then cropped
    normal_resized = normal_img.resize(size, Image.Resampling.LANCZOS)
    crop_box_resized = (
        size[0] // 4,
        size[1] // 4,
        3 * size[0] // 4,
        3 * size[1] // 4
    )
    normal_crop = normal_resized.crop(crop_box_resized)

    # Pneumonia - resized then cropped
    pneumonia_resized = pneumonia_img.resize(size, Image.Resampling.LANCZOS)
    pneumonia_crop = pneumonia_resized.crop(crop_box_resized)

    # Plot Normal
    axes[0, idx].imshow(normal_crop, cmap='gray')
    axes[0, idx].set_title(f'NORMAL\n{size[0]}×{size[1]}', fontsize=10, fontweight='bold', color='green')
    axes[0, idx].axis('off')

    # Plot Pneumonia
    axes[1, idx].imshow(pneumonia_crop, cmap='gray')
    axes[1, idx].set_title(f'PNEUMONIA\n{size[0]}×{size[1]}', fontsize=10, fontweight='bold', color='red')
    axes[1, idx].axis('off')

plt.tight_layout()
plt.savefig('resolution_comparison_zoomed.png', dpi=150, bbox_inches='tight')
print("✓ Saved: resolution_comparison_zoomed.png")

# ==============================================================================
# SIDE-BY-SIDE COMPARISON
# ==============================================================================

print("\n[4] Creating side-by-side comparison...")

# Create a 2x2 comparison for easier viewing
fig3, axes = plt.subplots(2, 2, figsize=(14, 14))
fig3.suptitle('Which Size Best Preserves Medical Information?', fontsize=16, fontweight='bold')

sizes_for_comparison = [(128, 128), (224, 224), (512, 512)]
selected_sizes = [(128, 128), (224, 224), (512, 512)]

positions = [(0, 0), (0, 1), (1, 0)]

for pos, size in zip(positions, selected_sizes):
    # Resize pneumonia image (more interesting to see patterns)
    resized = pneumonia_img.resize(size, Image.Resampling.LANCZOS)

    # Calculate info
    original_pixels = normal_img.size[0] * normal_img.size[1]
    resized_pixels = size[0] * size[1]
    percentage = (resized_pixels / original_pixels) * 100

    axes[pos].imshow(resized, cmap='gray')
    axes[pos].set_title(f'{size[0]}×{size[1]} pixels\n({percentage:.2f}% of original)\nCan you see lung patterns?',
                        fontsize=12, fontweight='bold')
    axes[pos].axis('off')

# Add original in bottom right
axes[1, 1].imshow(pneumonia_img, cmap='gray')
axes[1, 1].set_title(f'ORIGINAL\n{pneumonia_img.size[0]}×{pneumonia_img.size[1]} pixels\n(100% - for reference)',
                     fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('resolution_comparison_sidebyside.png', dpi=150, bbox_inches='tight')
print("✓ Saved: resolution_comparison_sidebyside.png")

# ==============================================================================
# ANALYSIS QUESTIONS
# ==============================================================================

print("\n" + "=" * 70)
print(" LOOK AT THE IMAGES AND ANSWER THESE QUESTIONS:")
print("=" * 70)

print("""
📊 Three images have been generated:
   1. resolution_comparison.png - Full overview
   2. resolution_comparison_zoomed.png - Lung region detail
   3. resolution_comparison_sidebyside.png - Easy comparison

Open these images and carefully observe them.

THEN, answer these questions:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION 1: Visual Inspection
Look at the PNEUMONIA images at different sizes.
Can you still see the cloudy/white patterns in the lungs at:
  • 64×64?    Yes / No
  • 128×128?  Yes / No  
  • 224×224?  Yes / No
  • 512×512?  Yes / No

QUESTION 2: Diagnostic Capability
Imagine you're a doctor. At which size can you CONFIDENTLY 
tell the difference between NORMAL and PNEUMONIA?
  • My answer: _______×_______

QUESTION 3: Trade-offs
Complete this sentence:
"I think ___×___ is the best choice because:
- Pattern visibility: _______________
- Training speed: _______________
- Detail preserved: _______________"

QUESTION 4: Your Concern
You said: "even 224×224 patterns not visible"
After looking at the actual images, do you still think this?
What changed your mind (or confirmed your concern)?

QUESTION 5: Surprising Discovery
What surprised you most about this comparison?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 THINK ABOUT:
- The difference between what humans need to see vs what CNNs need
- CNNs can detect patterns we can't always consciously see
- Many research papers achieve 90%+ accuracy with 224×224

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# ==============================================================================
# TECHNICAL INFO
# ==============================================================================

print("\n" + "=" * 70)
print(" TECHNICAL INFORMATION")
print("=" * 70)

print(f"""
Original Image Statistics:
  Size: {normal_img.size[0]}×{normal_img.size[1]} pixels
  Total pixels: {normal_img.size[0] * normal_img.size[1]:,}
  File format: JPEG (compressed)

Resized Image Statistics:
""")

for size in sizes_to_test:
    original_pixels = normal_img.size[0] * normal_img.size[1]
    resized_pixels = size[0] * size[1]
    percentage = (resized_pixels / original_pixels) * 100
    reduction = 100 - percentage

    print(f"  {size[0]}×{size[1]}:")
    print(f"    - Total pixels: {resized_pixels:,}")
    print(f"    - Percentage kept: {percentage:.2f}%")
    print(f"    - Information reduction: {reduction:.2f}%")
    print(f"    - Approximate training time: {get_training_time_estimate(size)}")
    print()


