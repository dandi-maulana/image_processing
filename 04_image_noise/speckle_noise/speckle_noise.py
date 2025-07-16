#!/usr/bin/env python3
"""
Speckle Noise Generator
Menghasilkan derau Speckle pada gambar
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

def add_speckle_noise(image_path, variance=0.1):
    """
    Menambahkan Speckle noise ke gambar
    
    Args:
        image_path: Path ke gambar input
        variance: Varian dari speckle noise
    """
    print(f"✨ SPECKLE NOISE GENERATOR")
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
    
    # Generate Speckle noise
    print(f"🔄 Generating Speckle noise (variance={variance})...")
    
    # Normalize to 0-1 range
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Generate multiplicative noise
    # Speckle noise: I_noisy = I * (1 + noise)
    noise = np.random.normal(0, variance**0.5, img_normalized.shape)
    
    # Apply speckle noise (multiplicative)
    noisy_normalized = img_normalized * (1 + noise)
    
    # Clip to valid range and convert back
    noisy_normalized = np.clip(noisy_normalized, 0, 1)
    noisy_img = (noisy_normalized * 255).astype(np.uint8)
    
    # Calculate the actual noise added
    noise_added = noisy_img.astype(np.float32) - img_rgb.astype(np.float32)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save noisy image
    output_path = f"speckle_noise_{base_name}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))
    
    # Save noise pattern
    noise_normalized = ((noise_added - noise_added.min()) / (noise_added.max() - noise_added.min()) * 255).astype(np.uint8)
    noise_path = f"speckle_noise_pattern_{base_name}.jpg"
    cv2.imwrite(noise_path, noise_normalized)
    
    # Save multiplicative noise pattern
    mult_noise_vis = ((noise - noise.min()) / (noise.max() - noise.min()) * 255).astype(np.uint8)
    mult_noise_path = f"speckle_multiplicative_pattern_{base_name}.jpg"
    cv2.imwrite(mult_noise_path, mult_noise_vis)
    
    # Create comparison plot
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(mult_noise_vis)
    plt.title(f'Multiplicative Noise\n(Variance: {variance})')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(noisy_img)
    plt.title('Image + Speckle Noise')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(noise_normalized)
    plt.title('Resulting Noise Pattern')
    plt.axis('off')
    
    # Plot noise histogram
    plt.subplot(2, 3, 5)
    plt.hist(noise.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.title('Multiplicative Noise Distribution')
    plt.xlabel('Noise Value')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(noise), color='red', linestyle='--', label=f'Mean: {np.mean(noise):.3f}')
    plt.axvline(0, color='green', linestyle='-', label='Zero')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot signal vs noise relationship
    plt.subplot(2, 3, 6)
    # Sample pixels for scatter plot
    sample_size = min(10000, img_normalized.size)
    sample_indices = np.random.choice(img_normalized.size, sample_size, replace=False)
    
    original_flat = img_normalized.flatten()[sample_indices]
    noise_flat = noise_added.flatten()[sample_indices]
    
    plt.scatter(original_flat * 255, noise_flat, alpha=0.3, s=1, color='orange')
    plt.xlabel('Original Pixel Value')
    plt.ylabel('Noise Added')
    plt.title('Signal vs Noise Relationship')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"speckle_comparison_{base_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Show results
    print(f"✅ Hasil disimpan:")
    print(f"   📁 {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    print(f"   📁 {noise_path} ({os.path.getsize(noise_path)/1024:.1f} KB)")
    print(f"   📁 {mult_noise_path} ({os.path.getsize(mult_noise_path)/1024:.1f} KB)")
    print(f"   📁 {plot_path} ({os.path.getsize(plot_path)/1024:.1f} KB)")
    
    # Show noise statistics
    print(f"\n📊 Noise Statistics:")
    print(f"   • Multiplicative noise mean: {np.mean(noise):.4f}")
    print(f"   • Multiplicative noise std: {np.std(noise):.4f}")
    print(f"   • Multiplicative noise variance: {np.var(noise):.4f}")
    print(f"   • Expected variance: {variance}")
    print(f"   • Resulting noise mean: {np.mean(noise_added):.2f}")
    print(f"   • Resulting noise std: {np.std(noise_added):.2f}")
    print(f"   • Signal-dependent: Ya (karakteristik Speckle)")

def main():
    """Main function"""
    print("🎯 Speckle Noise Generator")
    
    # Konfigurasi - ubah sesuai kebutuhan
    image_path = "/home/nugraha/Pictures/nugraha_05.jpeg"  # Ganti dengan path gambar Anda
    variance = 0.1    # Varian noise
    
    # Generate noise
    add_speckle_noise(image_path, variance)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()