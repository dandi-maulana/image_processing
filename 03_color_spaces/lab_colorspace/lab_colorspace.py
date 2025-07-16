#!/usr/bin/env python3
"""
LAB Color Space Analysis
Analisis ruang warna LAB dan aplikasinya
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
    """Membuat gambar test untuk LAB"""
    print("🎨 Membuat gambar test untuk LAB color space...")
    
    # Buat gambar dengan variasi warna dan pencahayaan
    img = np.zeros((500, 700, 3), dtype=np.uint8)
    
    # Gradasi warna dengan pencahayaan berbeda
    colors = [
        [255, 0, 0],    # Red
        [255, 127, 0],  # Orange
        [255, 255, 0],  # Yellow
        [0, 255, 0],    # Green
        [0, 255, 255],  # Cyan
        [0, 0, 255],    # Blue
        [127, 0, 255],  # Violet
        [255, 0, 255],  # Magenta
    ]
    
    # Buat gradasi pencahayaan untuk setiap warna
    for i, color in enumerate(colors):
        x_start = i * 80
        for j in range(80):
            # Variasi brightness
            brightness = j / 80.0
            adjusted_color = [int(c * brightness) for c in color]
            cv2.rectangle(img, (x_start + j, 50), (x_start + j + 1, 150), adjusted_color, -1)
    
    # Gradasi abu-abu untuk luminance analysis
    for i in range(300):
        gray_val = int(255 * (i / 300))
        cv2.rectangle(img, (200 + i, 200), (201 + i, 250), [gray_val, gray_val, gray_val], -1)
    
    # Warna kulit (skin tones) untuk skin detection
    skin_colors = [
        [255, 219, 172],  # Light skin
        [241, 194, 125],  # Medium light skin
        [224, 172, 105],  # Medium skin
        [198, 134, 66],   # Medium dark skin
        [141, 85, 36],    # Dark skin
    ]
    
    for i, skin_color in enumerate(skin_colors):
        cv2.rectangle(img, (100 + i * 60, 300), (150 + i * 60, 350), skin_color, -1)
    
    # Objek dengan warna berbeda untuk color difference analysis
    cv2.circle(img, (150, 400), 30, [255, 100, 100], -1)  # Light red
    cv2.circle(img, (250, 400), 30, [200, 100, 100], -1)  # Darker red
    cv2.circle(img, (350, 400), 30, [100, 255, 100], -1)  # Light green
    cv2.circle(img, (450, 400), 30, [100, 200, 100], -1)  # Darker green
    
    # Gradasi saturasi
    for i in range(200):
        saturation = i / 200.0
        color = [int(255 * saturation), int(100 * saturation), int(200 * saturation)]
        cv2.rectangle(img, (450, 50 + i), (500, 51 + i), color, -1)
    
    # Noise untuk denoising test
    noise_region = img[350:450, 500:600].copy()
    noise = np.random.normal(0, 20, noise_region.shape).astype(np.int16)
    noise_region = np.clip(noise_region.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img[350:450, 500:600] = noise_region
    
    cv2.imwrite("colorspace_test.jpg", img)
    print("✅ Test image created: colorspace_test.jpg")
    return "colorspace_test.jpg"

def analyze_lab_colorspace(image_path):
    """
    Analisis ruang warna LAB
    
    Args:
        image_path: Path ke gambar input
    """
    print(f"🔬 LAB COLOR SPACE ANALYSIS")
    print(f"=" * 50)
    
    # Load image
    if not os.path.exists(image_path):
        print(f"❌ File tidak ditemukan: {image_path}")
        image_path = create_test_image()
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Tidak dapat membaca gambar: {image_path}")
        return
    
    # Convert BGR to RGB dan LAB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    print(f"📖 Gambar dimuat: {image_path}")
    print(f"📏 Ukuran: {img_rgb.shape}")
    
    # Pisahkan channel LAB
    l_channel = img_lab[:, :, 0]  # Lightness (0-255)
    a_channel = img_lab[:, :, 1]  # Green-Red axis (0-255, 128=neutral)
    b_channel = img_lab[:, :, 2]  # Blue-Yellow axis (0-255, 128=neutral)
    
    # Normalisasi A dan B channels untuk visualisasi
    a_normalized = ((a_channel.astype(np.float32) - 128) / 128 * 127 + 128).astype(np.uint8)
    b_normalized = ((b_channel.astype(np.float32) - 128) / 128 * 127 + 128).astype(np.uint8)
    
    # Buat visualisasi masing-masing channel
    l_vis = np.zeros_like(img_lab)
    a_vis = np.zeros_like(img_lab)
    b_vis = np.zeros_like(img_lab)
    
    # L channel (grayscale)
    l_vis[:, :, 0] = l_channel
    l_vis[:, :, 1] = 128  # Neutral A
    l_vis[:, :, 2] = 128  # Neutral B
    
    # A channel (Green-Red)
    a_vis[:, :, 0] = l_channel  # Keep lightness
    a_vis[:, :, 1] = a_channel
    a_vis[:, :, 2] = 128  # Neutral B
    
    # B channel (Blue-Yellow)
    b_vis[:, :, 0] = l_channel  # Keep lightness
    b_vis[:, :, 1] = 128  # Neutral A
    b_vis[:, :, 2] = b_channel
    
    # Convert back to RGB for visualization
    l_vis_rgb = cv2.cvtColor(l_vis, cv2.COLOR_LAB2RGB)
    a_vis_rgb = cv2.cvtColor(a_vis, cv2.COLOR_LAB2RGB)
    b_vis_rgb = cv2.cvtColor(b_vis, cv2.COLOR_LAB2RGB)
    
    # Color enhancement using LAB
    # Enhance contrast in L channel
    l_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l_channel)
    
    # Create enhanced image
    img_enhanced = img_lab.copy()
    img_enhanced[:, :, 0] = l_enhanced
    img_enhanced_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)
    
    # Color correction example
    # Adjust A and B channels for color balance
    img_corrected = img_lab.copy().astype(np.float32)
    img_corrected[:, :, 1] = img_corrected[:, :, 1] * 0.9  # Reduce red
    img_corrected[:, :, 2] = img_corrected[:, :, 2] * 1.1  # Increase yellow
    img_corrected = np.clip(img_corrected, 0, 255).astype(np.uint8)
    img_corrected_rgb = cv2.cvtColor(img_corrected, cv2.COLOR_LAB2RGB)
    
    # Skin detection using LAB
    # Skin typically has specific ranges in LAB space
    skin_mask = cv2.inRange(img_lab, 
                           np.array([20, 133, 133]),   # Lower bound
                           np.array([255, 173, 173]))  # Upper bound
    
    skin_result = cv2.bitwise_and(img_rgb, img_rgb, mask=skin_mask)
    
    # Color difference calculation (Delta E)
    def calculate_delta_e(lab1, lab2):
        """Calculate Delta E (color difference) between two LAB colors"""
        dl = lab1[0] - lab2[0]
        da = lab1[1] - lab2[1]
        db = lab1[2] - lab2[2]
        return np.sqrt(dl**2 + da**2 + db**2)
    
    # Calculate color differences between regions
    region1 = img_lab[400:430, 120:180]  # Red circle area
    region2 = img_lab[400:430, 220:280]  # Darker red circle area
    
    avg_color1 = np.mean(region1.reshape(-1, 3), axis=0)
    avg_color2 = np.mean(region2.reshape(-1, 3), axis=0)
    
    delta_e = calculate_delta_e(avg_color1, avg_color2)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save individual channels
    cv2.imwrite(f"lab_l_channel_{base_name}.jpg", l_channel)
    cv2.imwrite(f"lab_a_channel_{base_name}.jpg", a_normalized)
    cv2.imwrite(f"lab_b_channel_{base_name}.jpg", b_normalized)
    
    # Save visualizations
    cv2.imwrite(f"lab_l_vis_{base_name}.jpg", cv2.cvtColor(l_vis_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"lab_a_vis_{base_name}.jpg", cv2.cvtColor(a_vis_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"lab_b_vis_{base_name}.jpg", cv2.cvtColor(b_vis_rgb, cv2.COLOR_RGB2BGR))
    
    # Save processed images
    cv2.imwrite(f"lab_enhanced_{base_name}.jpg", cv2.cvtColor(img_enhanced_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"lab_corrected_{base_name}.jpg", cv2.cvtColor(img_corrected_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"lab_skin_detection_{base_name}.jpg", cv2.cvtColor(skin_result, cv2.COLOR_RGB2BGR))
    
    # Create comprehensive analysis plot
    fig = plt.figure(figsize=(20, 16))
    
    # Row 1: Original and LAB channels
    plt.subplot(4, 4, 1)
    plt.imshow(img_rgb)
    plt.title('Original RGB Image')
    plt.axis('off')
    
    plt.subplot(4, 4, 2)
    plt.imshow(l_channel, cmap='gray')
    plt.title('L Channel (Lightness)')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(a_normalized, cmap='RdGy_r')
    plt.title('A Channel (Green-Red)')
    plt.axis('off')
    
    plt.subplot(4, 4, 4)
    plt.imshow(b_normalized, cmap='RdYlBu_r')
    plt.title('B Channel (Blue-Yellow)')
    plt.axis('off')
    
    # Row 2: Channel visualizations
    plt.subplot(4, 4, 5)
    plt.imshow(l_vis_rgb)
    plt.title('L Channel Visualization')
    plt.axis('off')
    
    plt.subplot(4, 4, 6)
    plt.imshow(a_vis_rgb)
    plt.title('A Channel Visualization')
    plt.axis('off')
    
    plt.subplot(4, 4, 7)
    plt.imshow(b_vis_rgb)
    plt.title('B Channel Visualization')
    plt.axis('off')
    
    # LAB color space representation
    plt.subplot(4, 4, 8)
    # Create A-B plane
    a_range = np.linspace(0, 255, 100)
    b_range = np.linspace(0, 255, 100)
    A, B = np.meshgrid(a_range, b_range)
    L = np.full_like(A, 128)  # Medium lightness
    
    lab_plane = np.stack([L, A, B], axis=2).astype(np.uint8)
    # Clip to valid LAB range
    lab_plane = np.clip(lab_plane, 0, 255)
    
    try:
        rgb_plane = cv2.cvtColor(lab_plane, cv2.COLOR_LAB2RGB)
        plt.imshow(rgb_plane, extent=[0, 255, 255, 0])
        plt.title('A-B Plane (L=128)')
        plt.xlabel('A (Green-Red)')
        plt.ylabel('B (Blue-Yellow)')
    except:
        plt.text(0.5, 0.5, 'LAB Plane\nVisualization', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('A-B Plane (L=128)')
    
    # Row 3: Applications
    plt.subplot(4, 4, 9)
    plt.imshow(img_enhanced_rgb)
    plt.title('Contrast Enhanced (L channel)')
    plt.axis('off')
    
    plt.subplot(4, 4, 10)
    plt.imshow(img_corrected_rgb)
    plt.title('Color Corrected')
    plt.axis('off')
    
    plt.subplot(4, 4, 11)
    plt.imshow(skin_result)
    plt.title('Skin Detection')
    plt.axis('off')
    
    plt.subplot(4, 4, 12)
    # Show color difference
    diff_img = np.zeros((100, 200, 3), dtype=np.uint8)
    diff_img[:, :100] = cv2.cvtColor(avg_color1.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB)
    diff_img[:, 100:] = cv2.cvtColor(avg_color2.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB)
    plt.imshow(diff_img)
    plt.title(f'Color Difference\nΔE = {delta_e:.2f}')
    plt.axis('off')
    
    # Row 4: Histograms and analysis
    plt.subplot(4, 4, 13)
    plt.hist(l_channel.flatten(), bins=50, color='gray', alpha=0.7)
    plt.title('L Channel Histogram')
    plt.xlabel('Lightness (0-255)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 14)
    plt.hist(a_channel.flatten(), bins=50, color='red', alpha=0.7)
    plt.title('A Channel Histogram')
    plt.xlabel('Green-Red (0-255)')
    plt.ylabel('Frequency')
    plt.axvline(x=128, color='black', linestyle='--', label='Neutral')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 15)
    plt.hist(b_channel.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('B Channel Histogram')
    plt.xlabel('Blue-Yellow (0-255)')
    plt.ylabel('Frequency')
    plt.axvline(x=128, color='black', linestyle='--', label='Neutral')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 4, 16)
    # LAB statistics
    channels = ['L', 'A', 'B']
    means = [np.mean(l_channel), np.mean(a_channel), np.mean(b_channel)]
    stds = [np.std(l_channel), np.std(a_channel), np.std(b_channel)]
    
    x = np.arange(len(channels))
    width = 0.35
    
    plt.bar(x - width/2, means, width, label='Mean', color=['gray', 'red', 'blue'], alpha=0.7)
    plt.bar(x + width/2, stds, width, label='Std Dev', color=['darkgray', 'darkred', 'darkblue'], alpha=0.7)
    plt.title('LAB Statistics')
    plt.xlabel('Channels')
    plt.ylabel('Value')
    plt.xticks(x, channels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"lab_analysis_{base_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create 3D LAB space visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Sample pixels for 3D plot
    sample_size = min(3000, img_lab.shape[0] * img_lab.shape[1])
    sample_indices = np.random.choice(img_lab.shape[0] * img_lab.shape[1], sample_size, replace=False)
    
    l_flat = l_channel.flatten()[sample_indices]
    a_flat = a_channel.flatten()[sample_indices]
    b_flat = b_channel.flatten()[sample_indices]
    
    # Convert to RGB for coloring
    lab_sample = np.column_stack([l_flat, a_flat, b_flat])
    lab_sample = lab_sample.reshape(-1, 1, 3).astype(np.uint8)
    
    try:
        rgb_sample = cv2.cvtColor(lab_sample, cv2.COLOR_LAB2RGB)
        rgb_sample = rgb_sample.reshape(-1, 3) / 255.0
        
        # 3D LAB space
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(l_flat, a_flat, b_flat, c=rgb_sample, s=2, alpha=0.6)
        ax.set_xlabel('L (Lightness)')
        ax.set_ylabel('A (Green-Red)')
        ax.set_zlabel('B (Blue-Yellow)')
        ax.set_title('LAB Color Space')
        
        # 2D AB projection
        ax2 = fig.add_subplot(122)
        ax2.scatter(a_flat, b_flat, c=rgb_sample, s=2, alpha=0.6)
        ax2.set_xlabel('A (Green-Red)')
        ax2.set_ylabel('B (Blue-Yellow)')
        ax2.set_title('A-B Plane Projection')
        ax2.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"⚠️ Warning: Could not create 3D visualization: {e}")
        plt.text(0.5, 0.5, 'LAB 3D Visualization\nNot Available', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plot_3d_path = f"lab_3d_analysis_{base_name}.png"
    plt.savefig(plot_3d_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Perceptual color difference analysis
    def analyze_color_differences():
        """Analyze perceptual color differences using LAB"""
        # Define test colors in LAB space
        test_colors = {
            'Red1': [50, 70, 50],
            'Red2': [50, 75, 50],
            'Green1': [50, -50, 50],
            'Green2': [50, -45, 50],
            'Blue1': [50, 20, -50],
            'Blue2': [50, 20, -45]
        }
        
        differences = {}
        for color1_name, color1 in test_colors.items():
            for color2_name, color2 in test_colors.items():
                if color1_name != color2_name:
                    delta_e = calculate_delta_e(color1, color2)
                    differences[f"{color1_name}-{color2_name}"] = delta_e
        
        return differences
    
    color_differences = analyze_color_differences()
    
    # Show results
    print(f"✅ Hasil disimpan:")
    print(f"   📁 lab_l_channel_{base_name}.jpg ({os.path.getsize(f'lab_l_channel_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 lab_a_channel_{base_name}.jpg ({os.path.getsize(f'lab_a_channel_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 lab_b_channel_{base_name}.jpg ({os.path.getsize(f'lab_b_channel_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 lab_enhanced_{base_name}.jpg ({os.path.getsize(f'lab_enhanced_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 lab_corrected_{base_name}.jpg ({os.path.getsize(f'lab_corrected_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 {plot_path} ({os.path.getsize(plot_path)/1024:.1f} KB)")
    print(f"   📁 {plot_3d_path} ({os.path.getsize(plot_3d_path)/1024:.1f} KB)")
    
    # Show statistics
    print(f"\n📊 LAB Statistics:")
    print(f"   • L channel - Mean: {np.mean(l_channel):.1f}, Std: {np.std(l_channel):.1f}")
    print(f"   • A channel - Mean: {np.mean(a_channel):.1f}, Std: {np.std(a_channel):.1f}")
    print(f"   • B channel - Mean: {np.mean(b_channel):.1f}, Std: {np.std(b_channel):.1f}")
    print(f"   • Color difference (ΔE): {delta_e:.2f}")
    
    # Color difference interpretation
    print(f"\n🔍 Color Difference Analysis:")
    if delta_e < 1:
        print(f"   • ΔE < 1: Tidak terlihat oleh mata manusia")
    elif delta_e < 3:
        print(f"   • 1 ≤ ΔE < 3: Sedikit terlihat oleh mata terlatih")
    elif delta_e < 6:
        print(f"   • 3 ≤ ΔE < 6: Terlihat jelas")
    else:
        print(f"   • ΔE ≥ 6: Sangat berbeda")
    
    # Skin detection results
    skin_pixels = np.sum(skin_mask > 0)
    total_pixels = skin_mask.shape[0] * skin_mask.shape[1]
    
    print(f"\n👤 Skin Detection Results:")
    print(f"   • Skin pixels detected: {skin_pixels:,}")
    print(f"   • Skin ratio: {skin_pixels/total_pixels*100:.1f}%")
    
    print(f"\n🎯 Implementasi LAB:")
    print(f"   • Color correction dan white balancing")
    print(f"   • Perceptual color differences (ΔE)")
    print(f"   • Skin detection dalam computer vision")
    print(f"   • Image enhancement (contrast, brightness)")
    print(f"   • Color quantization dan palette generation")
    print(f"   • Professional photo editing")
    print(f"   • Quality control dalam printing industry")
    print(f"   • Uniform color space untuk scientific analysis")

def main():
    """Main function"""
    print("🎯 LAB Color Space Analysis")
    
    # Konfigurasi
    image_path = "/home/nugraha/Pictures/nugraha_03.jpeg"   # Ganti dengan path gambar Anda
    
    # Analyze LAB color space
    analyze_lab_colorspace(image_path)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()