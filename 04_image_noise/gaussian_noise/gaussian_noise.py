#!/usr/bin/env python3
"""
Gaussian Noise Generator
Menghasilkan derau Gaussian pada gambar
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

def add_gaussian_noise(image_path, mean=0, std=25):
    """
    Menambahkan Gaussian noise ke gambar
    
    Args:
        image_path: Path ke gambar input
        mean: Mean dari distribusi Gaussian (default: 0)
        std: Standard deviation (default: 25)
    """
    print(f"🔵 GAUSSIAN NOISE GENERATOR")
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
    
    # Generate Gaussian noise
    print(f"🔄 Generating Gaussian noise (μ={mean}, σ={std})...")
    noise = np.random.normal(mean, std, img_rgb.shape).astype(np.float32)
    
    # Add noise to image
    noisy_img = img_rgb.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save noisy image
    output_path = f"gaussian_noise_{base_name}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))
    
    # Save noise pattern
    noise_normalized = ((noise - noise.min()) / (noise.max() - noise.min()) * 255).astype(np.uint8)
    noise_path = f"gaussian_noise_pattern_{base_name}.jpg"
    cv2.imwrite(noise_path, noise_normalized)
    
    # Create comparison plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(noise_normalized)
    plt.title(f'Gaussian Noise Pattern\n(μ={mean}, σ={std})')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(noisy_img)
    plt.title('Image + Gaussian Noise')
    plt.axis('off')
    
    plt.tight_layout()
    plot_path = f"gaussian_comparison_{base_name}.png"
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

def main():
    """Main function"""
    print("🎯 Gaussian Noise Generator")
    
    # Konfigurasi - ubah sesuai kebutuhan
    image_path = "/home/nugraha/Pictures/nugraha_01.jpeg"  # Ganti dengan path gambar Anda
    mean = 0          # Mean noise
    std = 25          # Standard deviation
    
    # Generate noise
    add_gaussian_noise(image_path, mean, std)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()