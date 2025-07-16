#!/usr/bin/env python3
"""
Poisson Noise Generator
Menghasilkan derau Poisson pada gambar
"""

import cv2
import numpy as np
import os
from pathlib import Path

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_test_image():
    """Membuat gambar test"""
    print("🎨 Membuat gambar test...")
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Tambahkan shapes berwarna
    cv2.rectangle(img, (50, 50), (200, 150), (255, 100, 100), -1)
    cv2.rectangle(img, (250, 50), (400, 150), (100, 255, 100), -1)
    cv2.rectangle(img, (450, 50), (550, 150), (100, 100, 255), -1)
    
    # Tambahkan circles
    cv2.circle(img, (150, 250), 60, (255, 255, 100), -1)
    cv2.circle(img, (350, 250), 60, (255, 100, 255), -1)
    cv2.circle(img, (500, 250), 60, (100, 255, 255), -1)
    
    # Tambahkan gradient
    for i in range(600):
        for j in range(50):
            img[320 + j, i] = [i//3, (i+100)//4, (i*2)//5]
    
    cv2.imwrite("test_image.jpg", img)
    print("✅ Test image created: test_image.jpg")
    return "test_image.jpg"

def add_poisson_noise(image_path, scale=1.0):
    """
    Menambahkan Poisson noise ke gambar
    
    Args:
        image_path: Path ke gambar input
        scale: Faktor skala untuk intensitas noise
    """
    print(f"🎯 POISSON NOISE GENERATOR")
    print(f"=" * 40)
    
    # Load image
    if not os.path.exists(image_path):
        print(f"❌ File tidak ditemukan: {image_path}")
        image_path = create_test_image()
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Tidak dapat membaca gambar: {image_path}")
        return
    
    print(f"📖 Gambar dimuat: {image_path}")
    print(f"📏 Ukuran: {img.shape}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Generate Poisson noise
    print(f"🔄 Generating Poisson noise (scale={scale})...")
    
    # Normalize to 0-1 range
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Apply scaling
    img_scaled = img_normalized * scale
    
    # Apply Poisson noise
    # Poisson noise is signal-dependent
    noisy_scaled = np.random.poisson(img_scaled * 255) / 255.0
    
    # Convert back to 0-255 range
    noisy_img = np.clip(noisy_scaled * 255, 0, 255).astype(np.uint8)
    
    # Calculate noise (difference)
    noise = noisy_img.astype(np.float32) - img_rgb.astype(np.float32)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save noisy image
    output_path = f"poisson_noise_{base_name}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))
    
    # Save noise pattern
    noise_normalized = ((noise - noise.min()) / (noise.max() - noise.min()) * 255).astype(np.uint8)
    noise_path = f"poisson_noise_pattern_{base_name}.jpg"
    cv2.imwrite(noise_path, noise_normalized)
    
    # Create comparison plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(noise_normalized)
    plt.title(f'Poisson Noise Pattern\n(Scale: {scale})')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(noisy_img)
    plt.title('Image + Poisson Noise')
    plt.axis('off')
    
    # Plot noise histogram
    plt.subplot(2, 2, 4)
    plt.hist(noise.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title('Poisson Noise Distribution')
    plt.xlabel('Noise Value')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(noise), color='red', linestyle='--', label=f'Mean: {np.mean(noise):.2f}')
    plt.axvline(np.median(noise), color='blue', linestyle='--', label=f'Median: {np.median(noise):.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"poisson_comparison_{base_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Show results
    print(f"✅ Hasil disimpan:")
    print(f"   📁 {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    print(f"   📁 {noise_path} ({os.path.getsize(noise_path)/1024:.1f} KB)")
    print(f"   📁 {plot_path} ({os.path.getsize(plot_path)/1024:.1f} KB)")
    
    # Show noise statistics
    print(f"\n📊 Noise Statistics:")
    print(f"   • Mean: {np.mean(noise):.2f}")
    print(f"   • Std: {np.std(noise):.2f}")
    print(f"   • Min: {np.min(noise):.2f}")
    print(f"   • Max: {np.max(noise):.2f}")
    print(f"   • Skewness: {calculate_skewness(noise):.2f}")
    print(f"   • Signal-dependent: Ya (karakteristik Poisson)")

def calculate_skewness(data):
    """Calculate skewness of data"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)

def main():
    """Main function"""
    print("🎯 Poisson Noise Generator")
    
    # Konfigurasi - ubah sesuai kebutuhan
    image_path = "/home/nugraha/Pictures/nugraha_03.jpeg"  # Ganti dengan path gambar Anda
    scale = 1.0       # Faktor skala
    
    # Generate noise
    add_poisson_noise(image_path, scale)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()