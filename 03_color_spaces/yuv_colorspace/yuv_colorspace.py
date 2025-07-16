#!/usr/bin/env python3
"""
YUV/YCbCr Color Space Analysis
Analisis ruang warna YUV dan YCbCr serta aplikasinya
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
    """Membuat gambar test untuk YUV/YCbCr"""
    print("🎨 Membuat gambar test untuk YUV/YCbCr color space...")
    
    # Buat gambar untuk video/broadcast testing
    img = np.zeros((500, 700, 3), dtype=np.uint8)
    
    # Color bars (seperti pada test pattern TV)
    color_bars = [
        [255, 255, 255],  # White
        [255, 255, 0],    # Yellow
        [0, 255, 255],    # Cyan
        [0, 255, 0],      # Green
        [255, 0, 255],    # Magenta
        [255, 0, 0],      # Red
        [0, 0, 255],      # Blue
        [0, 0, 0],        # Black
    ]
    
    # Buat color bars
    bar_width = 700 // len(color_bars)
    for i, color in enumerate(color_bars):
        cv2.rectangle(img, (i * bar_width, 50), ((i + 1) * bar_width, 150), color, -1)
    
    # Gradasi luminance (brightness)
    for i in range(400):
        brightness = int(255 * (i / 400))
        cv2.rectangle(img, (150 + i, 200), (151 + i, 250), [brightness, brightness, brightness], -1)
    
    # Gradasi chrominance
    # U component (Blue-Yellow)
    for i in range(200):
        u_val = int(255 * (i / 200))
        # Simulate YUV to RGB conversion effect
        color = [128, 128, u_val]
        cv2.rectangle(img, (100 + i, 300), (101 + i, 350), color, -1)
    
    # V component (Red-Cyan)
    for i in range(200):
        v_val = int(255 * (i / 200))
        # Simulate YUV to RGB conversion effect
        color = [v_val, 128, 128]
        cv2.rectangle(img, (100 + i, 370), (101 + i, 420), color, -1)
    
    # Skin tone samples (important for broadcast)
    skin_tones = [
        [255, 220, 177],  # Light skin
        [241, 194, 125],  # Medium light
        [224, 172, 105],  # Medium
        [198, 134, 66],   # Medium dark
        [141, 85, 36],    # Dark
        [87, 58, 42],     # Very dark
    ]
    
    for i, skin_tone in enumerate(skin_tones):
        cv2.rectangle(img, (400 + i * 40, 300), (435 + i * 40, 350), skin_tone, -1)
    
    # Test patterns untuk compression
    # High frequency pattern
    for i in range(50):
        for j in range(50):
            if (i + j) % 4 < 2:
                cv2.rectangle(img, (500 + i * 3, 200 + j * 3), (503 + i * 3, 203 + j * 3), [255, 255, 255], -1)
            else:
                cv2.rectangle(img, (500 + i * 3, 200 + j * 3), (503 + i * 3, 203 + j * 3), [0, 0, 0], -1)
    
    # Low frequency pattern
    for i in range(25):
        for j in range(25):
            if (i + j) % 4 < 2:
                cv2.rectangle(img, (500 + i * 6, 350 + j * 6), (506 + i * 6, 356 + j * 6), [200, 100, 100], -1)
            else:
                cv2.rectangle(img, (500 + i * 6, 350 + j * 6), (506 + i * 6, 356 + j * 6), [100, 200, 100], -1)
    
    cv2.imwrite("colorspace_test.jpg", img)
    print("✅ Test image created: colorspace_test.jpg")
    return "colorspace_test.jpg"

def rgb_to_yuv(rgb):
    """Convert RGB to YUV manually"""
    # ITU-R BT.601 standard
    Y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    U = -0.147 * rgb[0] - 0.289 * rgb[1] + 0.436 * rgb[2]
    V = 0.615 * rgb[0] - 0.515 * rgb[1] - 0.100 * rgb[2]
    return np.array([Y, U, V])

def yuv_to_rgb(yuv):
    """Convert YUV to RGB manually"""
    R = yuv[0] + 1.140 * yuv[2]
    G = yuv[0] - 0.394 * yuv[1] - 0.581 * yuv[2]
    B = yuv[0] + 2.032 * yuv[1]
    return np.array([R, G, B])

def analyze_yuv_colorspace(image_path):
    """
    Analisis ruang warna YUV/YCbCr
    
    Args:
        image_path: Path ke gambar input
    """
    print(f"📺 YUV/YCbCr COLOR SPACE ANALYSIS")
    print(f"=" * 50)
    
    # Load image
    if not os.path.exists(image_path):
        print(f"❌ File tidak ditemukan: {image_path}")
        image_path = create_test_image()
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Tidak dapat membaca gambar: {image_path}")
        return
    
    # Convert BGR to RGB dan YUV
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    print(f"📖 Gambar dimuat: {image_path}")
    print(f"📏 Ukuran: {img_rgb.shape}")
    
    # Pisahkan channel YUV
    y_channel = img_yuv[:, :, 0]  # Luminance
    u_channel = img_yuv[:, :, 1]  # Chrominance U (Blue-Yellow)
    v_channel = img_yuv[:, :, 2]  # Chrominance V (Red-Cyan)
    
    # Pisahkan channel YCrCb
    y_ycrcb = img_ycrcb[:, :, 0]  # Luminance
    cr_channel = img_ycrcb[:, :, 1]  # Chrominance Cr (Red)
    cb_channel = img_ycrcb[:, :, 2]  # Chrominance Cb (Blue)
    
    # Buat visualisasi masing-masing channel
    y_vis = np.zeros_like(img_yuv)
    u_vis = np.zeros_like(img_yuv)
    v_vis = np.zeros_like(img_yuv)
    
    # Y channel (grayscale)
    y_vis[:, :, 0] = y_channel
    y_vis[:, :, 1] = 128  # Neutral U
    y_vis[:, :, 2] = 128  # Neutral V
    
    # U channel
    u_vis[:, :, 0] = 128  # Neutral Y
    u_vis[:, :, 1] = u_channel
    u_vis[:, :, 2] = 128  # Neutral V
    
    # V channel
    v_vis[:, :, 0] = 128  # Neutral Y
    v_vis[:, :, 1] = 128  # Neutral U
    v_vis[:, :, 2] = v_channel
    
    # Convert back to RGB for visualization
    y_vis_rgb = cv2.cvtColor(y_vis, cv2.COLOR_YUV2RGB)
    u_vis_rgb = cv2.cvtColor(u_vis, cv2.COLOR_YUV2RGB)
    v_vis_rgb = cv2.cvtColor(v_vis, cv2.COLOR_YUV2RGB)
    
    # Simulate chroma subsampling (4:2:0)
    def chroma_subsample_420(img_yuv):
        """Simulate 4:2:0 chroma subsampling"""
        h, w = img_yuv.shape[:2]
        subsampled = img_yuv.copy()
        
        # Subsample U and V channels by 2x2
        for i in range(0, h-1, 2):
            for j in range(0, w-1, 2):
                # Average 2x2 block
                u_avg = np.mean(img_yuv[i:i+2, j:j+2, 1])
                v_avg = np.mean(img_yuv[i:i+2, j:j+2, 2])
                
                # Apply to 2x2 block
                subsampled[i:i+2, j:j+2, 1] = u_avg
                subsampled[i:i+2, j:j+2, 2] = v_avg
        
        return subsampled
    
    # Apply chroma subsampling
    img_420 = chroma_subsample_420(img_yuv)
    img_420_rgb = cv2.cvtColor(img_420, cv2.COLOR_YUV2RGB)
    
    # Simulate different chroma subsampling ratios
    def chroma_subsample_422(img_yuv):
        """Simulate 4:2:2 chroma subsampling"""
        h, w = img_yuv.shape[:2]
        subsampled = img_yuv.copy()
        
        # Subsample U and V channels horizontally by 2
        for i in range(h):
            for j in range(0, w-1, 2):
                u_avg = np.mean(img_yuv[i, j:j+2, 1])
                v_avg = np.mean(img_yuv[i, j:j+2, 2])
                
                subsampled[i, j:j+2, 1] = u_avg
                subsampled[i, j:j+2, 2] = v_avg
        
        return subsampled
    
    img_422 = chroma_subsample_422(img_yuv)
    img_422_rgb = cv2.cvtColor(img_422, cv2.COLOR_YUV2RGB)
    
    # Quantization simulation (for compression)
    def quantize_channel(channel, levels=32):
        """Quantize channel to specific levels"""
        quantized = np.round(channel / 255 * (levels - 1)) / (levels - 1) * 255
        return quantized.astype(np.uint8)
    
    # Apply quantization to chrominance channels
    img_quantized = img_yuv.copy()
    img_quantized[:, :, 1] = quantize_channel(img_yuv[:, :, 1], 16)  # U channel
    img_quantized[:, :, 2] = quantize_channel(img_yuv[:, :, 2], 16)  # V channel
    img_quantized_rgb = cv2.cvtColor(img_quantized, cv2.COLOR_YUV2RGB)
    
    # Skin detection using YCrCb
    skin_mask = cv2.inRange(img_ycrcb, 
                           np.array([0, 133, 77]),    # Lower bound
                           np.array([255, 173, 127])) # Upper bound
    
    skin_result = cv2.bitwise_and(img_rgb, img_rgb, mask=skin_mask)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save individual channels
    cv2.imwrite(f"yuv_y_channel_{base_name}.jpg", y_channel)
    cv2.imwrite(f"yuv_u_channel_{base_name}.jpg", u_channel)
    cv2.imwrite(f"yuv_v_channel_{base_name}.jpg", v_channel)
    
    # Save visualizations
    cv2.imwrite(f"yuv_y_vis_{base_name}.jpg", cv2.cvtColor(y_vis_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"yuv_u_vis_{base_name}.jpg", cv2.cvtColor(u_vis_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"yuv_v_vis_{base_name}.jpg", cv2.cvtColor(v_vis_rgb, cv2.COLOR_RGB2BGR))
    
    # Save compression simulations
    cv2.imwrite(f"yuv_420_subsampled_{base_name}.jpg", cv2.cvtColor(img_420_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"yuv_422_subsampled_{base_name}.jpg", cv2.cvtColor(img_422_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"yuv_quantized_{base_name}.jpg", cv2.cvtColor(img_quantized_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"yuv_skin_detection_{base_name}.jpg", cv2.cvtColor(skin_result, cv2.COLOR_RGB2BGR))
    
    # Create comprehensive analysis plot
    fig = plt.figure(figsize=(20, 20))
    
    # Row 1: Original and YUV channels
    plt.subplot(5, 4, 1)
    plt.imshow(img_rgb)
    plt.title('Original RGB Image')
    plt.axis('off')
    
    plt.subplot(5, 4, 2)
    plt.imshow(y_channel, cmap='gray')
    plt.title('Y Channel (Luminance)')
    plt.axis('off')
    
    plt.subplot(5, 4, 3)
    plt.imshow(u_channel, cmap='RdYlBu')
    plt.title('U Channel (Blue-Yellow)')
    plt.axis('off')
    
    plt.subplot(5, 4, 4)
    plt.imshow(v_channel, cmap='RdGy')
    plt.title('V Channel (Red-Cyan)')
    plt.axis('off')
    
    # Row 2: YCrCb channels
    plt.subplot(5, 4, 5)
    plt.imshow(y_ycrcb, cmap='gray')
    plt.title('Y Channel (YCrCb)')
    plt.axis('off')
    
    plt.subplot(5, 4, 6)
    plt.imshow(cr_channel, cmap='Reds')
    plt.title('Cr Channel (Red)')
    plt.axis('off')
    
    plt.subplot(5, 4, 7)
    plt.imshow(cb_channel, cmap='Blues')
    plt.title('Cb Channel (Blue)')
    plt.axis('off')
    
    plt.subplot(5, 4, 8)
    plt.imshow(skin_result)
    plt.title('Skin Detection (YCrCb)')
    plt.axis('off')
    
    # Row 3: Channel visualizations
    plt.subplot(5, 4, 9)
    plt.imshow(y_vis_rgb)
    plt.title('Y Channel Visualization')
    plt.axis('off')
    
    plt.subplot(5, 4, 10)
    plt.imshow(u_vis_rgb)
    plt.title('U Channel Visualization')
    plt.axis('off')
    
    plt.subplot(5, 4, 11)
    plt.imshow(v_vis_rgb)
    plt.title('V Channel Visualization')
    plt.axis('off')
    
    # YUV color space representation
    plt.subplot(5, 4, 12)
    # Create UV plane
    u_range = np.linspace(0, 255, 100)
    v_range = np.linspace(0, 255, 100)
    U, V = np.meshgrid(u_range, v_range)
    Y = np.full_like(U, 128)  # Medium luminance
    
    yuv_plane = np.stack([Y, U, V], axis=2).astype(np.uint8)
    
    try:
        rgb_plane = cv2.cvtColor(yuv_plane, cv2.COLOR_YUV2RGB)
        plt.imshow(rgb_plane, extent=[0, 255, 255, 0])
        plt.title('U-V Plane (Y=128)')
        plt.xlabel('U (Blue-Yellow)')
        plt.ylabel('V (Red-Cyan)')
    except:
        plt.text(0.5, 0.5, 'YUV Plane\nVisualization', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('U-V Plane (Y=128)')
    
    # Row 4: Compression effects
    plt.subplot(5, 4, 13)
    plt.imshow(img_rgb)
    plt.title('Original (4:4:4)')
    plt.axis('off')
    
    plt.subplot(5, 4, 14)
    plt.imshow(img_422_rgb)
    plt.title('4:2:2 Subsampling')
    plt.axis('off')
    
    plt.subplot(5, 4, 15)
    plt.imshow(img_420_rgb)
    plt.title('4:2:0 Subsampling')
    plt.axis('off')
    
    plt.subplot(5, 4, 16)
    plt.imshow(img_quantized_rgb)
    plt.title('Quantized Chrominance')
    plt.axis('off')
    
    # Row 5: Analysis and histograms
    plt.subplot(5, 4, 17)
    plt.hist(y_channel.flatten(), bins=50, color='gray', alpha=0.7)
    plt.title('Y Channel Histogram')
    plt.xlabel('Luminance (0-255)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(5, 4, 18)
    plt.hist(u_channel.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('U Channel Histogram')
    plt.xlabel('Blue-Yellow (0-255)')
    plt.ylabel('Frequency')
    plt.axvline(x=128, color='black', linestyle='--', label='Neutral')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(5, 4, 19)
    plt.hist(v_channel.flatten(), bins=50, color='red', alpha=0.7)
    plt.title('V Channel Histogram')
    plt.xlabel('Red-Cyan (0-255)')
    plt.ylabel('Frequency')
    plt.axvline(x=128, color='black', linestyle='--', label='Neutral')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(5, 4, 20)
    # Compression ratio analysis
    compression_methods = ['Original', '4:2:2', '4:2:0', 'Quantized']
    
    # Calculate MSE for compression quality
    mse_422 = np.mean((img_rgb.astype(np.float32) - img_422_rgb.astype(np.float32))**2)
    mse_420 = np.mean((img_rgb.astype(np.float32) - img_420_rgb.astype(np.float32))**2)
    mse_quantized = np.mean((img_rgb.astype(np.float32) - img_quantized_rgb.astype(np.float32))**2)
    
    mse_values = [0, mse_422, mse_420, mse_quantized]
    
    plt.bar(compression_methods, mse_values, color=['green', 'yellow', 'orange', 'red'])
    plt.title('Compression Quality (MSE)')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"yuv_analysis_{base_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Show results
    print(f"✅ Hasil disimpan:")
    print(f"   📁 yuv_y_channel_{base_name}.jpg ({os.path.getsize(f'yuv_y_channel_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 yuv_u_channel_{base_name}.jpg ({os.path.getsize(f'yuv_u_channel_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 yuv_v_channel_{base_name}.jpg ({os.path.getsize(f'yuv_v_channel_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 yuv_420_subsampled_{base_name}.jpg ({os.path.getsize(f'yuv_420_subsampled_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 yuv_422_subsampled_{base_name}.jpg ({os.path.getsize(f'yuv_422_subsampled_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 {plot_path} ({os.path.getsize(plot_path)/1024:.1f} KB)")
    
    # Show statistics
    print(f"\n📊 YUV Statistics:")
    print(f"   • Y channel - Mean: {np.mean(y_channel):.1f}, Std: {np.std(y_channel):.1f}")
    print(f"   • U channel - Mean: {np.mean(u_channel):.1f}, Std: {np.std(u_channel):.1f}")
    print(f"   • V channel - Mean: {np.mean(v_channel):.1f}, Std: {np.std(v_channel):.1f}")
    
    print(f"\n📊 Compression Analysis:")
    print(f"   • 4:2:2 MSE: {mse_422:.2f}")
    print(f"   • 4:2:0 MSE: {mse_420:.2f}")
    print(f"   • Quantized MSE: {mse_quantized:.2f}")
    
    # Skin detection results
    skin_pixels = np.sum(skin_mask > 0)
    total_pixels = skin_mask.shape[0] * skin_mask.shape[1]
    
    print(f"\n👤 Skin Detection Results:")
    print(f"   • Skin pixels detected: {skin_pixels:,}")
    print(f"   • Skin ratio: {skin_pixels/total_pixels*100:.1f}%")
    
    print(f"\n🎯 Implementasi YUV/YCrCb:")
    print(f"   • Video compression (MPEG, H.264)")
    print(f"   • JPEG image compression")
    print(f"   • Broadcast television (PAL, NTSC)")
    print(f"   • Chroma subsampling (4:4:4, 4:2:2, 4:2:0)")
    print(f"   • Skin detection dalam computer vision")
    print(f"   • Video streaming optimization")
    print(f"   • Digital video processing")
    print(f"   • Color space conversion untuk multimedia")

def main():
    """Main function"""
    print("🎯 YUV/YCrCb Color Space Analysis")
    
    # Konfigurasi
    image_path = "/home/nugraha/Pictures/nugraha_04.jpeg"   # Ganti dengan path gambar Anda
    
    # Analyze YUV/YCrCb color space
    analyze_yuv_colorspace(image_path)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()