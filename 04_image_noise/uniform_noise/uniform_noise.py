#!/usr/bin/env python3
"""
Uniform Noise Generator
Menghasilkan derau Uniform pada gambar
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

def add_uniform_noise(image_path, low=-30, high=30):
    """
    Menambahkan Uniform noise ke gambar
    
    Args:
        image_path: Path ke gambar input
        low: Batas bawah distribusi uniform
        high: Batas atas distribusi uniform
    """
    print(f"📊 UNIFORM NOISE GENERATOR")
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
    
    # Generate Uniform noise
    print(f"🔄 Generating Uniform noise (range: [{low}, {high}])...")
    noise = np.random.uniform(low, high, img_rgb.shape).astype(np.float32)
    
    # Add noise to image
    noisy_img = img_rgb.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save noisy image
    output_path = f"uniform_noise_{base_name}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))
    
    # Save noise pattern
    noise_normalized = ((noise - noise.min()) / (noise.max() - noise.min()) * 255).astype(np.uint8)
    noise_path = f"uniform_noise_pattern_{base_name}.jpg"
    cv2.imwrite(noise_path, noise_normalized)
    
    # Create comparison plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(noise_normalized)
    plt.title(f'Uniform Noise Pattern\n(Range: [{low}, {high}])')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(noisy_img)
    plt.title('Image + Uniform Noise')
    plt.axis('off')
    
    # Plot noise histogram
    plt.subplot(2, 2, 4)
    plt.hist(noise.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Uniform Noise Distribution')
    plt.xlabel('Noise Value')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(noise), color='red', linestyle='--', label=f'Mean: {np.mean(noise):.2f}')
    plt.axvline(low, color='green', linestyle='--', label=f'Min: {low}')
    plt.axvline(high, color='green', linestyle='--', label=f'Max: {high}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"uniform_comparison_{base_name}.png"
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
    print(f"   • Expected range: [{low}, {high}]")
    print(f"   • Theoretical mean: {(low + high) / 2:.2f}")

def main():
    """Main function"""
    print("🎯 Uniform Noise Generator")
    
    # Konfigurasi - ubah sesuai kebutuhan
    image_path = "/home/nugraha/Pictures/nugraha_06.jpeg"  # Ganti dengan path gambar Anda
    low = -30         # Batas bawah
    high = 30         # Batas atas
    
    # Generate noise
    add_uniform_noise(image_path, low, high)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()