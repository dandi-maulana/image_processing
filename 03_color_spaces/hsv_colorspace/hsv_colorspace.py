#!/usr/bin/env python3
"""
HSV Color Space Analysis
Analisis ruang warna HSV dan aplikasinya
"""

import cv2
import numpy as np
import os
from pathlib import Path

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_test_image():
    """Membuat gambar test dengan berbagai warna"""
    print("🎨 Membuat gambar test untuk HSV...")
    
    # Buat gambar dengan berbagai warna dan saturasi
    img = np.zeros((500, 700, 3), dtype=np.uint8)
    
    # Warna-warna dengan saturasi penuh
    colors = [
        ([255, 0, 0], "Red"),
        ([255, 127, 0], "Orange"),
        ([255, 255, 0], "Yellow"),
        ([127, 255, 0], "Yellow-Green"),
        ([0, 255, 0], "Green"),
        ([0, 255, 127], "Green-Cyan"),
        ([0, 255, 255], "Cyan"),
        ([0, 127, 255], "Light Blue"),
        ([0, 0, 255], "Blue"),
        ([127, 0, 255], "Blue-Violet"),
        ([255, 0, 255], "Magenta"),
        ([255, 0, 127], "Pink")
    ]
    
    # Buat color wheel
    for i, (color, name) in enumerate(colors):
        start_angle = i * 30
        end_angle = (i + 1) * 30
        cv2.ellipse(img, (200, 200), (150, 150), 0, start_angle, end_angle, color, -1)
    
    # Buat gradasi saturasi
    for i in range(200):
        saturation = int(255 * (i / 200))
        color = [saturation, saturation, 255]  # Blue dengan saturasi bervariasi
        cv2.rectangle(img, (450, 50 + i), (500, 51 + i), color, -1)
    
    # Buat gradasi value (brightness)
    for i in range(200):
        value = int(255 * (i / 200))
        color = [0, 0, value]  # Blue dengan brightness bervariasi
        cv2.rectangle(img, (520, 50 + i), (570, 51 + i), color, -1)
    
    # Buat gradasi hue
    for i in range(360):
        hue = i
        # Convert HSV to RGB
        hsv_color = np.uint8([[[hue//2, 255, 255]]])  # OpenCV uses 0-179 for hue
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
        x = int(600 + 80 * np.cos(np.radians(i)))
        y = int(200 + 80 * np.sin(np.radians(i)))
        cv2.circle(img, (x, y), 3, rgb_color.tolist(), -1)
    
    # Tambahkan objek untuk segmentasi warna
    cv2.circle(img, (150, 400), 40, [255, 0, 0], -1)    # Red circle
    cv2.circle(img, (250, 400), 40, [0, 255, 0], -1)    # Green circle
    cv2.circle(img, (350, 400), 40, [0, 0, 255], -1)    # Blue circle
    
    # Tambahkan gradasi abu-abu
    for i in range(200):
        gray = int(255 * (i / 200))
        cv2.rectangle(img, (450 + i, 350), (451 + i, 400), [gray, gray, gray], -1)
    
    cv2.imwrite("colorspace_test.jpg", img)
    print("✅ Test image created: colorspace_test.jpg")
    return "colorspace_test.jpg"

def analyze_hsv_colorspace(image_path):
    """
    Analisis ruang warna HSV
    
    Args:
        image_path: Path ke gambar input
    """
    print(f"🌈 HSV COLOR SPACE ANALYSIS")
    print(f"=" * 50)
    
    # Load image
    if not os.path.exists(image_path):
        print(f"❌ File tidak ditemukan: {image_path}")
        image_path = create_test_image()
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Tidak dapat membaca gambar: {image_path}")
        return
    
    # Convert BGR to RGB dan HSV
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    print(f"📖 Gambar dimuat: {image_path}")
    print(f"📏 Ukuran: {img_rgb.shape}")
    
    # Pisahkan channel HSV
    h_channel = img_hsv[:, :, 0]  # Hue (0-179)
    s_channel = img_hsv[:, :, 1]  # Saturation (0-255)
    v_channel = img_hsv[:, :, 2]  # Value (0-255)
    
    # Buat visualisasi masing-masing channel
    h_vis = np.zeros_like(img_hsv)
    s_vis = np.zeros_like(img_hsv)
    v_vis = np.zeros_like(img_hsv)
    
    # Hue visualization (colorful)
    h_vis[:, :, 0] = h_channel
    h_vis[:, :, 1] = 255  # Full saturation
    h_vis[:, :, 2] = 255  # Full value
    h_vis_rgb = cv2.cvtColor(h_vis, cv2.COLOR_HSV2RGB)
    
    # Saturation visualization
    s_vis[:, :, 0] = 120  # Fixed hue (green)
    s_vis[:, :, 1] = s_channel
    s_vis[:, :, 2] = 255  # Full value
    s_vis_rgb = cv2.cvtColor(s_vis, cv2.COLOR_HSV2RGB)
    
    # Value visualization
    v_vis[:, :, 0] = 120  # Fixed hue (green)
    v_vis[:, :, 1] = 255  # Full saturation
    v_vis[:, :, 2] = v_channel
    v_vis_rgb = cv2.cvtColor(v_vis, cv2.COLOR_HSV2RGB)
    
    # Color segmentation examples
    # Segmentasi warna merah
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    red_mask = mask_red1 + mask_red2
    
    # Segmentasi warna hijau
    lower_green = np.array([40, 120, 70])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
    
    # Segmentasi warna biru
    lower_blue = np.array([100, 120, 70])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
    
    # Apply masks
    red_result = cv2.bitwise_and(img_rgb, img_rgb, mask=red_mask)
    green_result = cv2.bitwise_and(img_rgb, img_rgb, mask=green_mask)
    blue_result = cv2.bitwise_and(img_rgb, img_rgb, mask=blue_mask)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save individual channels
    cv2.imwrite(f"hsv_hue_{base_name}.jpg", cv2.cvtColor(h_vis_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"hsv_saturation_{base_name}.jpg", cv2.cvtColor(s_vis_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"hsv_value_{base_name}.jpg", cv2.cvtColor(v_vis_rgb, cv2.COLOR_RGB2BGR))
    
    # Save segmentation results
    cv2.imwrite(f"hsv_red_segmentation_{base_name}.jpg", cv2.cvtColor(red_result, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"hsv_green_segmentation_{base_name}.jpg", cv2.cvtColor(green_result, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"hsv_blue_segmentation_{base_name}.jpg", cv2.cvtColor(blue_result, cv2.COLOR_RGB2BGR))
    
    # Create comprehensive analysis plot
    fig = plt.figure(figsize=(20, 16))
    
    # Row 1: Original and HSV channels
    plt.subplot(4, 4, 1)
    plt.imshow(img_rgb)
    plt.title('Original RGB Image')
    plt.axis('off')
    
    plt.subplot(4, 4, 2)
    plt.imshow(h_vis_rgb)
    plt.title('Hue Channel (0-179°)')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(s_vis_rgb)
    plt.title('Saturation Channel (0-255)')
    plt.axis('off')
    
    plt.subplot(4, 4, 4)
    plt.imshow(v_vis_rgb)
    plt.title('Value Channel (0-255)')
    plt.axis('off')
    
    # Row 2: Grayscale channels
    plt.subplot(4, 4, 5)
    plt.imshow(h_channel, cmap='hsv')
    plt.title('Hue (Grayscale)')
    plt.axis('off')
    
    plt.subplot(4, 4, 6)
    plt.imshow(s_channel, cmap='gray')
    plt.title('Saturation (Grayscale)')
    plt.axis('off')
    
    plt.subplot(4, 4, 7)
    plt.imshow(v_channel, cmap='gray')
    plt.title('Value (Grayscale)')
    plt.axis('off')
    
    # Color wheel
    plt.subplot(4, 4, 8)
    # Create color wheel
    y, x = np.ogrid[:200, :200]
    center = 100
    mask = (x - center) ** 2 + (y - center) ** 2 <= 90**2
    
    hue = np.arctan2(y - center, x - center)
    hue = (hue + np.pi) / (2 * np.pi) * 179
    saturation = np.sqrt((x - center) ** 2 + (y - center) ** 2) / 90 * 255
    value = np.ones_like(hue) * 255
    
    hsv_wheel = np.stack([hue, saturation, value], axis=2).astype(np.uint8)
    hsv_wheel[~mask] = 0
    rgb_wheel = cv2.cvtColor(hsv_wheel, cv2.COLOR_HSV2RGB)
    
    plt.imshow(rgb_wheel)
    plt.title('HSV Color Wheel')
    plt.axis('off')
    
    # Row 3: Color segmentation
    plt.subplot(4, 4, 9)
    plt.imshow(red_result)
    plt.title('Red Color Segmentation')
    plt.axis('off')
    
    plt.subplot(4, 4, 10)
    plt.imshow(green_result)
    plt.title('Green Color Segmentation')
    plt.axis('off')
    
    plt.subplot(4, 4, 11)
    plt.imshow(blue_result)
    plt.title('Blue Color Segmentation')
    plt.axis('off')
    
    # Combined segmentation
    plt.subplot(4, 4, 12)
    combined_mask = red_mask + green_mask + blue_mask
    combined_result = cv2.bitwise_and(img_rgb, img_rgb, mask=combined_mask)
    plt.imshow(combined_result)
    plt.title('Combined Color Segmentation')
    plt.axis('off')
    
    # Row 4: Histograms and analysis
    plt.subplot(4, 4, 13)
    plt.hist(h_channel.flatten(), bins=50, color='red', alpha=0.7, label='Hue')
    plt.title('Hue Histogram')
    plt.xlabel('Hue (0-179)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 14)
    plt.hist(s_channel.flatten(), bins=50, color='green', alpha=0.7, label='Saturation')
    plt.title('Saturation Histogram')
    plt.xlabel('Saturation (0-255)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 15)
    plt.hist(v_channel.flatten(), bins=50, color='blue', alpha=0.7, label='Value')
    plt.title('Value Histogram')
    plt.xlabel('Value (0-255)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 16)
    # HSV statistics
    channels = ['Hue', 'Saturation', 'Value']
    means = [np.mean(h_channel), np.mean(s_channel), np.mean(v_channel)]
    stds = [np.std(h_channel), np.std(s_channel), np.std(v_channel)]
    
    x = np.arange(len(channels))
    width = 0.35
    
    plt.bar(x - width/2, means, width, label='Mean', color=['red', 'green', 'blue'], alpha=0.7)
    plt.bar(x + width/2, stds, width, label='Std Dev', color=['darkred', 'darkgreen', 'darkblue'], alpha=0.7)
    plt.title('HSV Statistics')
    plt.xlabel('Channels')
    plt.ylabel('Value')
    plt.xticks(x, channels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"hsv_analysis_{base_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create 3D HSV space visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Sample pixels for 3D plot
    sample_size = min(3000, img_hsv.shape[0] * img_hsv.shape[1])
    sample_indices = np.random.choice(img_hsv.shape[0] * img_hsv.shape[1], sample_size, replace=False)
    
    h_flat = h_channel.flatten()[sample_indices]
    s_flat = s_channel.flatten()[sample_indices]
    v_flat = v_channel.flatten()[sample_indices]
    
    # Convert to RGB for coloring
    hsv_sample = np.column_stack([h_flat, s_flat, v_flat])
    hsv_sample = hsv_sample.reshape(-1, 1, 3).astype(np.uint8)
    rgb_sample = cv2.cvtColor(hsv_sample, cv2.COLOR_HSV2RGB)
    rgb_sample = rgb_sample.reshape(-1, 3) / 255.0
    
    # 3D HSV space (cylindrical coordinates)
    ax = fig.add_subplot(121, projection='3d')
    
    # Convert to cylindrical coordinates
    theta = h_flat * 2 * np.pi / 179  # Convert hue to radians
    r = s_flat / 255.0  # Normalize saturation
    z = v_flat / 255.0  # Normalize value
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    ax.scatter(x, y, z, c=rgb_sample, s=2, alpha=0.6)
    ax.set_xlabel('Saturation * cos(Hue)')
    ax.set_ylabel('Saturation * sin(Hue)')
    ax.set_zlabel('Value')
    ax.set_title('HSV Color Space (Cylindrical)')
    
    # 2D HS projection
    ax2 = fig.add_subplot(122)
    ax2.scatter(h_flat, s_flat, c=rgb_sample, s=2, alpha=0.6)
    ax2.set_xlabel('Hue')
    ax2.set_ylabel('Saturation')
    ax2.set_title('Hue-Saturation Projection')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_3d_path = f"hsv_3d_analysis_{base_name}.png"
    plt.savefig(plot_3d_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Color analysis by hue ranges
    hue_ranges = {
        'Red': [(0, 10), (170, 179)],
        'Orange': [(10, 25)],
        'Yellow': [(25, 35)],
        'Green': [(35, 85)],
        'Cyan': [(85, 95)],
        'Blue': [(95, 125)],
        'Violet': [(125, 155)],
        'Magenta': [(155, 170)]
    }
    
    color_stats = {}
    for color_name, ranges in hue_ranges.items():
        total_pixels = 0
        for hue_range in ranges:
            mask = (h_channel >= hue_range[0]) & (h_channel <= hue_range[1])
            # Only count pixels with sufficient saturation
            mask = mask & (s_channel > 50)
            total_pixels += np.sum(mask)
        color_stats[color_name] = total_pixels
    
    # Show results
    print(f"✅ Hasil disimpan:")
    print(f"   📁 hsv_hue_{base_name}.jpg ({os.path.getsize(f'hsv_hue_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 hsv_saturation_{base_name}.jpg ({os.path.getsize(f'hsv_saturation_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 hsv_value_{base_name}.jpg ({os.path.getsize(f'hsv_value_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 {plot_path} ({os.path.getsize(plot_path)/1024:.1f} KB)")
    print(f"   📁 {plot_3d_path} ({os.path.getsize(plot_3d_path)/1024:.1f} KB)")
    
    # Show statistics
    print(f"\n📊 HSV Statistics:")
    print(f"   • Hue - Mean: {np.mean(h_channel):.1f}°, Std: {np.std(h_channel):.1f}°")
    print(f"   • Saturation - Mean: {np.mean(s_channel):.1f}, Std: {np.std(s_channel):.1f}")
    print(f"   • Value - Mean: {np.mean(v_channel):.1f}, Std: {np.std(v_channel):.1f}")
    
    print(f"\n🎨 Color Distribution:")
    for color, count in color_stats.items():
        percentage = (count / (img_hsv.shape[0] * img_hsv.shape[1])) * 100
        print(f"   • {color}: {count:,} pixels ({percentage:.1f}%)")
    
    # Segmentation statistics
    red_pixels = np.sum(red_mask > 0)
    green_pixels = np.sum(green_mask > 0)
    blue_pixels = np.sum(blue_mask > 0)
    
    print(f"\n🔍 Segmentation Results:")
    print(f"   • Red objects: {red_pixels:,} pixels")
    print(f"   • Green objects: {green_pixels:,} pixels")
    print(f"   • Blue objects: {blue_pixels:,} pixels")
    
    print(f"\n🎯 Implementasi HSV:")
    print(f"   • Color-based object detection")
    print(f"   • Skin detection dan face recognition")
    print(f"   • Image segmentation berdasarkan warna")
    print(f"   • Chroma key (green screen) processing")
    print(f"   • Color filtering dan enhancement")
    print(f"   • Tracking objek berdasarkan warna")
    print(f"   • Artistic effects dan stylization")

def main():
    """Main function"""
    print("🎯 HSV Color Space Analysis")
    
    # Konfigurasi
    image_path = "/home/nugraha/Pictures/nugraha_02.jpeg"  # Ganti dengan path gambar Anda
    
    # Analyze HSV color space
    analyze_hsv_colorspace(image_path)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()