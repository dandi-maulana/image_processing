#!/usr/bin/env python3
"""
RGB Color Space Analysis
Analisis ruang warna RGB dan aplikasinya
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
    print("🎨 Membuat gambar test dengan berbagai warna...")
    
    # Buat gambar dengan berbagai warna
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Warna dasar RGB
    img[50:150, 50:150] = [255, 0, 0]      # Merah
    img[50:150, 200:300] = [0, 255, 0]     # Hijau
    img[50:150, 350:450] = [0, 0, 255]     # Biru
    
    # Warna campuran
    img[200:300, 50:150] = [255, 255, 0]   # Kuning (R+G)
    img[200:300, 200:300] = [255, 0, 255]  # Magenta (R+B)
    img[200:300, 350:450] = [0, 255, 255]  # Cyan (G+B)
    
    # Gradasi abu-abu
    for i in range(100):
        img[350:400, 50+i*5:55+i*5] = [i*2.55, i*2.55, i*2.55]
    
    # Warna kompleks
    img[50:150, 500:550] = [255, 128, 64]  # Orange
    img[200:300, 500:550] = [128, 64, 255] # Purple
    
    cv2.imwrite("colorspace_test.jpg", img)
    print("✅ Test image created: colorspace_test.jpg")
    return "colorspace_test.jpg"

def analyze_rgb_colorspace(image_path):
    """
    Analisis ruang warna RGB
    
    Args:
        image_path: Path ke gambar input
    """
    print(f"🔴🟢🔵 RGB COLOR SPACE ANALYSIS")
    print(f"=" * 50)
    
    # Load image
    if not os.path.exists(image_path):
        print(f"❌ File tidak ditemukan: {image_path}")
        image_path = create_test_image()
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Tidak dapat membaca gambar: {image_path}")
        return
    
    # Convert BGR to RGB (OpenCV uses BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"📖 Gambar dimuat: {image_path}")
    print(f"📏 Ukuran: {img_rgb.shape}")
    
    # Pisahkan channel RGB
    r_channel = img_rgb[:, :, 0]
    g_channel = img_rgb[:, :, 1]
    b_channel = img_rgb[:, :, 2]
    
    # Buat visualisasi masing-masing channel
    r_vis = np.zeros_like(img_rgb)
    g_vis = np.zeros_like(img_rgb)
    b_vis = np.zeros_like(img_rgb)
    
    r_vis[:, :, 0] = r_channel
    g_vis[:, :, 1] = g_channel
    b_vis[:, :, 2] = b_channel
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save individual channels
    cv2.imwrite(f"rgb_red_channel_{base_name}.jpg", cv2.cvtColor(r_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"rgb_green_channel_{base_name}.jpg", cv2.cvtColor(g_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"rgb_blue_channel_{base_name}.jpg", cv2.cvtColor(b_vis, cv2.COLOR_RGB2BGR))
    
    # Create comprehensive analysis plot
    fig = plt.figure(figsize=(20, 16))
    
    # Row 1: Original and RGB channels
    plt.subplot(4, 4, 1)
    plt.imshow(img_rgb)
    plt.title('Original RGB Image')
    plt.axis('off')
    
    plt.subplot(4, 4, 2)
    plt.imshow(r_channel, cmap='Reds')
    plt.title('Red Channel')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(g_channel, cmap='Greens')
    plt.title('Green Channel')
    plt.axis('off')
    
    plt.subplot(4, 4, 4)
    plt.imshow(b_channel, cmap='Blues')
    plt.title('Blue Channel')
    plt.axis('off')
    
    # Row 2: Channel visualizations
    plt.subplot(4, 4, 5)
    plt.imshow(r_vis)
    plt.title('Red Channel Visualization')
    plt.axis('off')
    
    plt.subplot(4, 4, 6)
    plt.imshow(g_vis)
    plt.title('Green Channel Visualization')
    plt.axis('off')
    
    plt.subplot(4, 4, 7)
    plt.imshow(b_vis)
    plt.title('Blue Channel Visualization')
    plt.axis('off')
    
    # Combined channels
    plt.subplot(4, 4, 8)
    combined = np.maximum(np.maximum(r_vis, g_vis), b_vis)
    plt.imshow(combined)
    plt.title('Combined Channels')
    plt.axis('off')
    
    # Row 3: Histograms
    plt.subplot(4, 4, 9)
    plt.hist(r_channel.flatten(), bins=50, color='red', alpha=0.7, label='Red')
    plt.hist(g_channel.flatten(), bins=50, color='green', alpha=0.7, label='Green')
    plt.hist(b_channel.flatten(), bins=50, color='blue', alpha=0.7, label='Blue')
    plt.title('RGB Histograms')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Individual histograms
    plt.subplot(4, 4, 10)
    plt.hist(r_channel.flatten(), bins=50, color='red', alpha=0.7)
    plt.title('Red Channel Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 11)
    plt.hist(g_channel.flatten(), bins=50, color='green', alpha=0.7)
    plt.title('Green Channel Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 12)
    plt.hist(b_channel.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Blue Channel Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Row 4: Analysis
    plt.subplot(4, 4, 13)
    # Channel statistics
    channels = ['Red', 'Green', 'Blue']
    means = [np.mean(r_channel), np.mean(g_channel), np.mean(b_channel)]
    stds = [np.std(r_channel), np.std(g_channel), np.std(b_channel)]
    
    x = np.arange(len(channels))
    width = 0.35
    
    plt.bar(x - width/2, means, width, label='Mean', color=['red', 'green', 'blue'], alpha=0.7)
    plt.bar(x + width/2, stds, width, label='Std Dev', color=['darkred', 'darkgreen', 'darkblue'], alpha=0.7)
    plt.title('Channel Statistics')
    plt.xlabel('Channels')
    plt.ylabel('Value')
    plt.xticks(x, channels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 14)
    # Color distribution
    colors = ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'White', 'Black']
    color_counts = []
    
    # Count dominant colors (simplified)
    red_dominant = np.sum((r_channel > 200) & (g_channel < 100) & (b_channel < 100))
    green_dominant = np.sum((r_channel < 100) & (g_channel > 200) & (b_channel < 100))
    blue_dominant = np.sum((r_channel < 100) & (g_channel < 100) & (b_channel > 200))
    yellow_dominant = np.sum((r_channel > 200) & (g_channel > 200) & (b_channel < 100))
    magenta_dominant = np.sum((r_channel > 200) & (g_channel < 100) & (b_channel > 200))
    cyan_dominant = np.sum((r_channel < 100) & (g_channel > 200) & (b_channel > 200))
    white_dominant = np.sum((r_channel > 200) & (g_channel > 200) & (b_channel > 200))
    black_dominant = np.sum((r_channel < 50) & (g_channel < 50) & (b_channel < 50))
    
    color_counts = [red_dominant, green_dominant, blue_dominant, yellow_dominant, 
                   magenta_dominant, cyan_dominant, white_dominant, black_dominant]
    
    plt.bar(range(len(colors)), color_counts, 
            color=['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white', 'black'])
    plt.title('Color Distribution')
    plt.xlabel('Colors')
    plt.ylabel('Pixel Count')
    plt.xticks(range(len(colors)), colors, rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 15)
    # Brightness analysis
    brightness = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
    plt.hist(brightness.flatten(), bins=50, color='gray', alpha=0.7)
    plt.title('Brightness Distribution')
    plt.xlabel('Brightness')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 16)
    # Saturation analysis (simplified)
    saturation = np.zeros_like(r_channel, dtype=np.float32)
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            rgb_max = max(r_channel[i,j], g_channel[i,j], b_channel[i,j])
            rgb_min = min(r_channel[i,j], g_channel[i,j], b_channel[i,j])
            if rgb_max != 0:
                saturation[i,j] = (rgb_max - rgb_min) / rgb_max
    
    plt.hist(saturation.flatten(), bins=50, color='purple', alpha=0.7)
    plt.title('Saturation Distribution')
    plt.xlabel('Saturation')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"rgb_analysis_{base_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create 3D RGB color space plot
    fig = plt.figure(figsize=(15, 10))
    
    # Sample pixels for 3D plot
    sample_size = min(5000, img_rgb.shape[0] * img_rgb.shape[1])
    sample_indices = np.random.choice(img_rgb.shape[0] * img_rgb.shape[1], sample_size, replace=False)
    
    r_flat = r_channel.flatten()[sample_indices]
    g_flat = g_channel.flatten()[sample_indices]
    b_flat = b_channel.flatten()[sample_indices]
    
    ax = fig.add_subplot(121, projection='3d')
    colors = np.column_stack([r_flat/255, g_flat/255, b_flat/255])
    ax.scatter(r_flat, g_flat, b_flat, c=colors, s=1, alpha=0.6)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('RGB Color Space Distribution')
    
    # 2D projections
    ax2 = fig.add_subplot(122)
    ax2.scatter(r_flat, g_flat, c=colors, s=1, alpha=0.6)
    ax2.set_xlabel('Red')
    ax2.set_ylabel('Green')
    ax2.set_title('RG Projection')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_3d_path = f"rgb_3d_analysis_{base_name}.png"
    plt.savefig(plot_3d_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Show results
    print(f"✅ Hasil disimpan:")
    print(f"   📁 rgb_red_channel_{base_name}.jpg ({os.path.getsize(f'rgb_red_channel_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 rgb_green_channel_{base_name}.jpg ({os.path.getsize(f'rgb_green_channel_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 rgb_blue_channel_{base_name}.jpg ({os.path.getsize(f'rgb_blue_channel_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 {plot_path} ({os.path.getsize(plot_path)/1024:.1f} KB)")
    print(f"   📁 {plot_3d_path} ({os.path.getsize(plot_3d_path)/1024:.1f} KB)")
    
    # Show statistics
    print(f"\n📊 RGB Statistics:")
    print(f"   • Red channel - Mean: {np.mean(r_channel):.1f}, Std: {np.std(r_channel):.1f}")
    print(f"   • Green channel - Mean: {np.mean(g_channel):.1f}, Std: {np.std(g_channel):.1f}")
    print(f"   • Blue channel - Mean: {np.mean(b_channel):.1f}, Std: {np.std(b_channel):.1f}")
    print(f"   • Brightness - Mean: {np.mean(brightness):.1f}, Std: {np.std(brightness):.1f}")
    print(f"   • Saturation - Mean: {np.mean(saturation):.3f}, Std: {np.std(saturation):.3f}")
    
    print(f"\n🎯 Implementasi RGB:")
    print(f"   • Monitor dan display devices")
    print(f"   • Web graphics dan HTML")
    print(f"   • Fotografi digital")
    print(f"   • Computer vision applications")
    print(f"   • Gaming dan multimedia")
    print(f"   • Additive color mixing")

def main():
    """Main function"""
    print("🎯 RGB Color Space Analysis")
    
    # Konfigurasi
    image_path = "/home/nugraha/Pictures/nugraha_01.jpeg"  # Ganti dengan path gambar Anda
    
    # Analyze RGB color space
    analyze_rgb_colorspace(image_path)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()