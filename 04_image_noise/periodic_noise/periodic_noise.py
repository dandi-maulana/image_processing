#!/usr/bin/env python3
"""
Periodic Noise Generator
Menghasilkan derau Periodic pada gambar
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

def add_periodic_noise(image_path, frequency=0.1, amplitude=30, pattern_type='sinusoidal'):
    """
    Menambahkan Periodic noise ke gambar
    
    Args:
        image_path: Path ke gambar input
        frequency: Frekuensi dari pola periodic
        amplitude: Amplitudo dari noise
        pattern_type: Jenis pola ('sinusoidal', 'square', 'diagonal')
    """
    print(f"🌊 PERIODIC NOISE GENERATOR")
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
    
    # Generate Periodic noise
    print(f"🔄 Generating Periodic noise (freq={frequency}, amp={amplitude}, pattern={pattern_type})...")
    
    height, width = img_rgb.shape[:2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Generate different periodic patterns
    if pattern_type == 'sinusoidal':
        # Sinusoidal pattern in both x and y directions
        pattern = amplitude * (np.sin(2 * np.pi * frequency * x / width) + 
                              np.sin(2 * np.pi * frequency * y / height))
    elif pattern_type == 'square':
        # Square wave pattern
        pattern = amplitude * (np.sign(np.sin(2 * np.pi * frequency * x / width)) + 
                              np.sign(np.sin(2 * np.pi * frequency * y / height)))
    elif pattern_type == 'diagonal':
        # Diagonal stripes
        pattern = amplitude * np.sin(2 * np.pi * frequency * (x + y) / (width + height))
    else:
        # Default to sinusoidal
        pattern = amplitude * (np.sin(2 * np.pi * frequency * x / width) + 
                              np.sin(2 * np.pi * frequency * y / height))
    
    # Expand pattern to 3 channels
    pattern_3d = np.stack([pattern] * 3, axis=2)
    
    # Add noise to image
    noisy_img = img_rgb.astype(np.float32) + pattern_3d
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save noisy image
    output_path = f"periodic_noise_{base_name}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))
    
    # Save noise pattern
    pattern_normalized = ((pattern - pattern.min()) / (pattern.max() - pattern.min()) * 255).astype(np.uint8)
    pattern_path = f"periodic_pattern_{base_name}.jpg"
    cv2.imwrite(pattern_path, pattern_normalized)
    
    # Create different frequency variants
    variants = []
    freq_variants = [frequency * 0.5, frequency * 2, frequency * 4]
    
    for i, freq in enumerate(freq_variants):
        if pattern_type == 'sinusoidal':
            variant_pattern = amplitude * (np.sin(2 * np.pi * freq * x / width) + 
                                          np.sin(2 * np.pi * freq * y / height))
        elif pattern_type == 'square':
            variant_pattern = amplitude * (np.sign(np.sin(2 * np.pi * freq * x / width)) + 
                                          np.sign(np.sin(2 * np.pi * freq * y / height)))
        elif pattern_type == 'diagonal':
            variant_pattern = amplitude * np.sin(2 * np.pi * freq * (x + y) / (width + height))
        else:
            variant_pattern = amplitude * (np.sin(2 * np.pi * freq * x / width) + 
                                          np.sin(2 * np.pi * freq * y / height))
        
        variant_3d = np.stack([variant_pattern] * 3, axis=2)
        variant_img = img_rgb.astype(np.float32) + variant_3d
        variant_img = np.clip(variant_img, 0, 255).astype(np.uint8)
        variants.append(variant_img)
    
    # Create comprehensive comparison plot
    plt.figure(figsize=(20, 12))
    
    # Row 1: Original, pattern, and result
    plt.subplot(3, 4, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(pattern_normalized, cmap='gray')
    plt.title(f'Periodic Pattern\n({pattern_type}, freq={frequency})')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(noisy_img)
    plt.title('Image + Periodic Noise')
    plt.axis('off')
    
    # Pattern analysis
    plt.subplot(3, 4, 4)
    plt.plot(pattern[height//2, :])
    plt.title('Horizontal Cross-section')
    plt.xlabel('X Position')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Row 2: Frequency variants
    for i, (freq, variant) in enumerate(zip(freq_variants, variants)):
        plt.subplot(3, 4, 5 + i)
        plt.imshow(variant)
        plt.title(f'Frequency = {freq:.2f}')
        plt.axis('off')
    
    # 2D Pattern visualization
    plt.subplot(3, 4, 8)
    plt.imshow(pattern, cmap='coolwarm')
    plt.title('2D Pattern Visualization')
    plt.colorbar()
    plt.axis('off')
    
    # Row 3: Fourier analysis
    plt.subplot(3, 4, 9)
    # FFT of the pattern
    fft_pattern = np.fft.fft2(pattern)
    fft_magnitude = np.log(np.abs(fft_pattern) + 1)
    fft_shifted = np.fft.fftshift(fft_magnitude)
    plt.imshow(fft_shifted, cmap='hot')
    plt.title('FFT of Pattern')
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.hist(pattern.flatten(), bins=50, alpha=0.7, color='cyan', edgecolor='black')
    plt.title('Pattern Distribution')
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 11)
    plt.plot(pattern[:, width//2])
    plt.title('Vertical Cross-section')
    plt.xlabel('Y Position')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 12)
    # Show different pattern types
    diagonal_pattern = amplitude * np.sin(2 * np.pi * frequency * (x + y) / (width + height))
    plt.imshow(diagonal_pattern, cmap='viridis')
    plt.title('Diagonal Pattern Example')
    plt.axis('off')
    
    plt.tight_layout()
    plot_path = f"periodic_comparison_{base_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Show results
    print(f"✅ Hasil disimpan:")
    print(f"   📁 {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    print(f"   📁 {pattern_path} ({os.path.getsize(pattern_path)/1024:.1f} KB)")
    print(f"   📁 {plot_path} ({os.path.getsize(plot_path)/1024:.1f} KB)")
    
    # Show noise statistics
    print(f"\n📊 Noise Statistics:")
    print(f"   • Pattern type: {pattern_type}")
    print(f"   • Frequency: {frequency}")
    print(f"   • Amplitude: {amplitude}")
    print(f"   • Pattern mean: {np.mean(pattern):.2f}")
    print(f"   • Pattern std: {np.std(pattern):.2f}")
    print(f"   • Pattern min: {np.min(pattern):.2f}")
    print(f"   • Pattern max: {np.max(pattern):.2f}")
    print(f"   • Periodic: Ya (karakteristik Periodic)")

def main():
    """Main function"""
    print("🎯 Periodic Noise Generator")
    
    # Konfigurasi - ubah sesuai kebutuhan
    image_path = "/home/nugraha/Pictures/nugraha_02.jpeg"  # Ganti dengan path gambar Anda
    frequency = 0.1       # Frekuensi pola
    amplitude = 30        # Amplitudo noise
    pattern_type = 'sinusoidal'  # 'sinusoidal', 'square', 'diagonal'
    
    # Generate noise
    add_periodic_noise(image_path, frequency, amplitude, pattern_type)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()