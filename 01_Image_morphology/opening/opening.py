#!/usr/bin/env python3
"""
Opening - Operasi Morfologi
Menghilangkan noise kecil sambil mempertahankan bentuk objek
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
    """Membuat gambar test untuk morfologi"""
    print("🎨 Membuat gambar test untuk operasi Opening...")
    
    # Buat gambar binary dengan objek utama dan noise
    img = np.zeros((400, 600), dtype=np.uint8)
    
    # Objek utama
    cv2.rectangle(img, (50, 50), (200, 150), 255, -1)      # Persegi besar
    cv2.circle(img, (350, 100), 60, 255, -1)               # Lingkaran besar
    cv2.ellipse(img, (500, 100), (80, 50), 0, 0, 360, 255, -1)  # Elips
    
    # Tambahkan noise kecil di sekitar objek
    noise_points = [(30, 30), (25, 170), (180, 30), (220, 170), 
                   (300, 50), (400, 50), (320, 170), (380, 170),
                   (450, 50), (550, 50), (480, 170), (520, 170)]
    
    for point in noise_points:
        cv2.circle(img, point, 3, 255, -1)
    
    # Tambahkan noise berupa garis tipis
    cv2.line(img, (50, 200), (200, 200), 255, 1)
    cv2.line(img, (250, 200), (400, 200), 255, 2)
    
    # Objek dengan protrusi kecil
    cv2.rectangle(img, (100, 250), (300, 350), 255, -1)    # Objek utama
    # Tambahkan protrusi kecil
    cv2.rectangle(img, (120, 240), (125, 250), 255, -1)
    cv2.rectangle(img, (200, 240), (205, 250), 255, -1)
    cv2.rectangle(img, (280, 240), (285, 250), 255, -1)
    cv2.rectangle(img, (120, 350), (125, 360), 255, -1)
    cv2.rectangle(img, (200, 350), (205, 360), 255, -1)
    cv2.rectangle(img, (280, 350), (285, 360), 255, -1)
    
    # Objek dengan noise internal
    cv2.rectangle(img, (400, 250), (550, 350), 255, -1)
    # Tambahkan beberapa pixel hitam (noise internal)
    cv2.circle(img, (450, 280), 2, 0, -1)
    cv2.circle(img, (480, 300), 2, 0, -1)
    cv2.circle(img, (510, 320), 2, 0, -1)
    
    cv2.imwrite("morphology_test.jpg", img)
    print("✅ Test image created: morphology_test.jpg")
    return "morphology_test.jpg"

def apply_opening(image_path, kernel_size=5, iterations=1):
    """
    Menerapkan operasi Opening pada gambar
    
    Args:
        image_path: Path ke gambar input
        kernel_size: Ukuran kernel (structuring element)
        iterations: Jumlah iterasi opening
    """
    print(f"🟢 OPENING - Morphological Operation")
    print(f"=" * 50)
    
    # Load image
    if not os.path.exists(image_path):
        print(f"❌ File tidak ditemukan: {image_path}")
        image_path = create_test_image()
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Tidak dapat membaca gambar: {image_path}")
        return
    
    print(f"📖 Gambar dimuat: {image_path}")
    print(f"📏 Ukuran: {img.shape}")
    
    # Convert to binary if not already
    if len(np.unique(img)) > 2:
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        print("🔄 Converted to binary image")
    
    # Create structuring element (kernel)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    print(f"🔧 Kernel size: {kernel_size}x{kernel_size}, Iterations: {iterations}")
    
    # Apply opening (erosion followed by dilation)
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    # Manual opening for comparison
    eroded = cv2.erode(img, kernel, iterations=iterations)
    manual_opened = cv2.dilate(eroded, kernel, iterations=iterations)
    
    # Test with different kernel sizes
    kernels = [3, 5, 7, 9]
    opening_results = []
    
    for k_size in kernels:
        k = np.ones((k_size, k_size), np.uint8)
        opened_temp = cv2.morphologyEx(img, cv2.MORPH_OPEN, k, iterations=1)
        opening_results.append(opened_temp)
    
    # Test with different kernel shapes
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    opened_cross = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_cross, iterations=iterations)
    opened_ellipse = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_ellipse, iterations=iterations)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save main result
    output_path = f"opening_{base_name}.jpg"
    cv2.imwrite(output_path, opened)
    
    # Save intermediate steps
    cv2.imwrite(f"opening_eroded_{base_name}.jpg", eroded)
    cv2.imwrite(f"opening_manual_{base_name}.jpg", manual_opened)
    cv2.imwrite(f"opening_cross_{base_name}.jpg", opened_cross)
    cv2.imwrite(f"opening_ellipse_{base_name}.jpg", opened_ellipse)
    
    # Create comparison plot
    plt.figure(figsize=(20, 15))
    
    # Row 1: Original and process steps
    plt.subplot(4, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(4, 4, 2)
    plt.imshow(eroded, cmap='gray')
    plt.title('Step 1: Erosion')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(opened, cmap='gray')
    plt.title('Step 2: Dilation (Opening)')
    plt.axis('off')
    
    # Show removed noise
    removed_noise = cv2.absdiff(img, opened)
    plt.subplot(4, 4, 4)
    plt.imshow(removed_noise, cmap='hot')
    plt.title('Removed Noise')
    plt.axis('off')
    
    # Row 2: Different kernel sizes
    for i, (k_size, result) in enumerate(zip(kernels, opening_results)):
        plt.subplot(4, 4, 5 + i)
        plt.imshow(result, cmap='gray')
        plt.title(f'Kernel {k_size}x{k_size}')
        plt.axis('off')
    
    # Row 3: Different kernel shapes
    plt.subplot(4, 4, 9)
    plt.imshow(kernel_cross, cmap='gray')
    plt.title('Cross Kernel')
    plt.axis('off')
    
    plt.subplot(4, 4, 10)
    plt.imshow(opened_cross, cmap='gray')
    plt.title('Opening with Cross')
    plt.axis('off')
    
    plt.subplot(4, 4, 11)
    plt.imshow(kernel_ellipse, cmap='gray')
    plt.title('Ellipse Kernel')
    plt.axis('off')
    
    plt.subplot(4, 4, 12)
    plt.imshow(opened_ellipse, cmap='gray')
    plt.title('Opening with Ellipse')
    plt.axis('off')
    
    # Row 4: Analysis
    plt.subplot(4, 4, 13)
    # Multiple iterations
    iter_results = []
    for iter_num in range(1, 6):
        iter_result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iter_num)
        iter_results.append(iter_result)
    
    plt.imshow(iter_results[2], cmap='gray')
    plt.title('3 Iterations')
    plt.axis('off')
    
    plt.subplot(4, 4, 14)
    plt.imshow(iter_results[4], cmap='gray')
    plt.title('5 Iterations')
    plt.axis('off')
    
    # Statistical analysis
    plt.subplot(4, 4, 15)
    original_pixels = np.sum(img == 255)
    opened_pixels = np.sum(opened == 255)
    pixel_counts = [original_pixels]
    
    for result in opening_results:
        pixel_counts.append(np.sum(result == 255))
    
    plt.bar(range(len(pixel_counts)), pixel_counts, color=['blue', 'red', 'green', 'orange', 'purple'])
    plt.title('White Pixels Count')
    plt.xlabel('Kernel Size')
    plt.ylabel('Pixel Count')
    plt.xticks(range(len(pixel_counts)), ['Original'] + [f'{k}x{k}' for k in kernels])
    plt.grid(True, alpha=0.3)
    
    # Iteration analysis
    plt.subplot(4, 4, 16)
    iter_pixel_counts = []
    for result in iter_results:
        iter_pixel_counts.append(np.sum(result == 255))
    
    plt.plot(range(1, 6), iter_pixel_counts, 'o-', color='green', linewidth=2)
    plt.title('Pixels vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('White Pixels')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"opening_analysis_{base_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Show results
    print(f"✅ Hasil disimpan:")
    print(f"   📁 {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    print(f"   📁 opening_eroded_{base_name}.jpg ({os.path.getsize(f'opening_eroded_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 opening_manual_{base_name}.jpg ({os.path.getsize(f'opening_manual_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 {plot_path} ({os.path.getsize(plot_path)/1024:.1f} KB)")
    
    # Show statistics
    noise_removed = original_pixels - opened_pixels
    print(f"\n📊 Opening Statistics:")
    print(f"   • Original white pixels: {original_pixels:,}")
    print(f"   • Opened white pixels: {opened_pixels:,}")
    print(f"   • Noise removed: {noise_removed:,}")
    print(f"   • Noise reduction: {noise_removed/original_pixels*100:.1f}%")
    
    # Compare manual vs built-in opening
    manual_pixels = np.sum(manual_opened == 255)
    print(f"   • Manual opening pixels: {manual_pixels:,}")
    print(f"   • Built-in vs Manual difference: {abs(opened_pixels - manual_pixels)} pixels")
    
    print(f"\n🎯 Kegunaan Opening:")
    print(f"   • Menghilangkan noise kecil")
    print(f"   • Mempertahankan bentuk objek utama")
    print(f"   • Memisahkan objek yang terhubung tipis")
    print(f"   • Smoothing kontur objek")
    print(f"   • Erosion diikuti Dilation")

def main():
    """Main function"""
    print("🎯 Opening - Morphological Operation")
    
    # Konfigurasi - ubah sesuai kebutuhan
    image_path = "/home/nugraha/Pictures/nugraha_09.jpeg"  # Ganti dengan path gambar Anda
    kernel_size = 5    # Ukuran kernel
    iterations = 1     # Jumlah iterasi
    
    # Apply opening
    apply_opening(image_path, kernel_size, iterations)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()