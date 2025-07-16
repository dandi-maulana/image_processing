#!/usr/bin/env python3
"""
Salt and Pepper Noise Generator
Menghasilkan derau Salt and Pepper pada gambar
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

def add_salt_pepper_noise(image_path, salt_prob=0.01, pepper_prob=0.01):
    """
    Menambahkan Salt and Pepper noise ke gambar
    
    Args:
        image_path: Path ke gambar input
        salt_prob: Probabilitas salt noise (pixel putih)
        pepper_prob: Probabilitas pepper noise (pixel hitam)
    """
    print(f"⚪⚫ SALT & PEPPER NOISE GENERATOR")
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
    
    # Generate Salt and Pepper noise
    print(f"🔄 Generating Salt & Pepper noise (salt={salt_prob*100:.1f}%, pepper={pepper_prob*100:.1f}%)...")
    
    noisy_img = img_rgb.copy()
    random_matrix = np.random.random(img_rgb.shape[:2])
    
    # Create noise mask for visualization
    noise_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    
    # Salt noise (white pixels)
    salt_mask = random_matrix < salt_prob
    noisy_img[salt_mask] = 255
    noise_mask[salt_mask] = 255  # White in mask
    
    # Pepper noise (black pixels)
    pepper_mask = random_matrix > (1 - pepper_prob)
    noisy_img[pepper_mask] = 0
    noise_mask[pepper_mask] = 128  # Gray in mask
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save noisy image
    output_path = f"salt_pepper_noise_{base_name}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))
    
    # Save noise pattern
    noise_path = f"salt_pepper_pattern_{base_name}.jpg"
    cv2.imwrite(noise_path, noise_mask)
    
    # Create comparison plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(noise_mask, cmap='gray')
    plt.title(f'Salt & Pepper Pattern\n(Salt={salt_prob*100:.1f}%, Pepper={pepper_prob*100:.1f}%)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(noisy_img)
    plt.title('Image + Salt & Pepper Noise')
    plt.axis('off')
    
    plt.tight_layout()
    plot_path = f"salt_pepper_comparison_{base_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate statistics
    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
    salt_count = np.sum(salt_mask)
    pepper_count = np.sum(pepper_mask)
    
    # Show results
    print(f"✅ Hasil disimpan:")
    print(f"   📁 {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    print(f"   📁 {noise_path} ({os.path.getsize(noise_path)/1024:.1f} KB)")
    print(f"   📁 {plot_path} ({os.path.getsize(plot_path)/1024:.1f} KB)")
    
    # Show noise statistics
    print(f"\n📊 Noise Statistics:")
    print(f"   • Total pixels: {total_pixels:,}")
    print(f"   • Salt pixels: {salt_count:,} ({salt_count/total_pixels*100:.3f}%)")
    print(f"   • Pepper pixels: {pepper_count:,} ({pepper_count/total_pixels*100:.3f}%)")
    print(f"   • Affected pixels: {salt_count + pepper_count:,} ({(salt_count + pepper_count)/total_pixels*100:.3f}%)")

def main():
    """Main function"""
    print("🎯 Salt & Pepper Noise Generator")
    
    # Konfigurasi - ubah sesuai kebutuhan
    image_path = "/home/nugraha/Pictures/nugraha_04.jpeg"  # Ganti dengan path gambar Anda
    salt_prob = 0.01      # Probabilitas salt noise (1%)
    pepper_prob = 0.01    # Probabilitas pepper noise (1%)
    
    # Generate noise
    add_salt_pepper_noise(image_path, salt_prob, pepper_prob)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()