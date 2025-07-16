#!/usr/bin/env python3
"""
Dilation - Operasi Morfologi
Memperbesar objek dan mengisi lubang kecil
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
    print("🎨 Membuat gambar test untuk operasi Dilation...")
    
    # Buat gambar binary dengan berbagai bentuk
    img = np.zeros((400, 600), dtype=np.uint8)
    
    # Tambahkan bentuk dengan lubang kecil
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)      # Persegi
    cv2.rectangle(img, (80, 80), (120, 120), 0, -1)        # Lubang di persegi
    
    # Lingkaran dengan lubang
    cv2.circle(img, (250, 100), 50, 255, -1)               # Lingkaran
    cv2.circle(img, (250, 100), 20, 0, -1)                 # Lubang di lingkaran
    
    # Bentuk yang terpisah dekat
    cv2.rectangle(img, (400, 50), (450, 100), 255, -1)     # Kotak kiri
    cv2.rectangle(img, (460, 50), (510, 100), 255, -1)     # Kotak kanan (dekat)
    
    # Garis putus-putus
    for i in range(50, 550, 20):
        cv2.rectangle(img, (i, 200), (i+10, 210), 255, -1)
    
    # Bentuk kecil yang terpisah
    cv2.circle(img, (100, 250), 8, 255, -1)
    cv2.circle(img, (120, 250), 8, 255, -1)
    cv2.circle(img, (140, 250), 8, 255, -1)
    
    # Bentuk dengan noise internal
    cv2.rectangle(img, (300, 230), (500, 280), 255, -1)
    # Tambahkan beberapa lubang kecil
    cv2.circle(img, (350, 250), 3, 0, -1)
    cv2.circle(img, (400, 255), 4, 0, -1)
    cv2.circle(img, (450, 250), 3, 0, -1)
    
    # Tambahkan teks dengan karakter terpisah
    cv2.putText(img, 'DILATION', (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    cv2.imwrite("morphology_test.jpg", img)
    print("✅ Test image created: morphology_test.jpg")
    return "morphology_test.jpg"

def apply_dilation(image_path, kernel_size=5, iterations=1):
    """
    Menerapkan operasi Dilation pada gambar
    
    Args:
        image_path: Path ke gambar input
        kernel_size: Ukuran kernel (structuring element)
        iterations: Jumlah iterasi dilation
    """
    print(f"🔵 DILATION - Morphological Operation")
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
    
    # Apply dilation
    dilated = cv2.dilate(img, kernel, iterations=iterations)
    
    # Test with different kernel sizes
    kernels = [3, 5, 7, 9]
    dilation_results = []
    
    for k_size in kernels:
        k = np.ones((k_size, k_size), np.uint8)
        dilated_temp = cv2.dilate(img, k, iterations=1)
        dilation_results.append(dilated_temp)
    
    # Test with different kernel shapes
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    dilated_cross = cv2.dilate(img, kernel_cross, iterations=iterations)
    dilated_ellipse = cv2.dilate(img, kernel_ellipse, iterations=iterations)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save main result
    output_path = f"dilation_{base_name}.jpg"
    cv2.imwrite(output_path, dilated)
    
    # Save other shapes
    cv2.imwrite(f"dilation_cross_{base_name}.jpg", dilated_cross)
    cv2.imwrite(f"dilation_ellipse_{base_name}.jpg", dilated_ellipse)
    
    # Create comparison plot
    plt.figure(figsize=(20, 15))
    
    # Row 1: Original and main result
    plt.subplot(4, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(4, 4, 2)
    plt.imshow(dilated, cmap='gray')
    plt.title(f'Dilation (kernel={kernel_size}x{kernel_size})')
    plt.axis('off')
    
    # Show kernel
    plt.subplot(4, 4, 3)
    plt.imshow(kernel, cmap='gray')
    plt.title(f'Rectangular Kernel\n({kernel_size}x{kernel_size})')
    plt.axis('off')
    
    # Show difference (added pixels)
    difference = cv2.absdiff(dilated, img)
    plt.subplot(4, 4, 4)
    plt.imshow(difference, cmap='hot')
    plt.title('Added Pixels (Difference)')
    plt.axis('off')
    
    # Row 2: Different kernel sizes
    for i, (k_size, result) in enumerate(zip(kernels, dilation_results)):
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
    plt.imshow(dilated_cross, cmap='gray')
    plt.title('Dilation with Cross')
    plt.axis('off')
    
    plt.subplot(4, 4, 11)
    plt.imshow(kernel_ellipse, cmap='gray')
    plt.title('Ellipse Kernel')
    plt.axis('off')
    
    plt.subplot(4, 4, 12)
    plt.imshow(dilated_ellipse, cmap='gray')
    plt.title('Dilation with Ellipse')
    plt.axis('off')
    
    # Row 4: Analysis
    plt.subplot(4, 4, 13)
    # Multiple iterations
    iter_results = []
    for iter_num in range(1, 6):
        iter_result = cv2.dilate(img, kernel, iterations=iter_num)
        iter_results.append(iter_result)
    
    # Show iteration 3 as example
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
    dilated_pixels = np.sum(dilated == 255)
    pixel_counts = [original_pixels]
    
    for result in dilation_results:
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
    
    plt.plot(range(1, 6), iter_pixel_counts, 'o-', color='blue', linewidth=2)
    plt.title('Pixels vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('White Pixels')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"dilation_analysis_{base_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Show results
    print(f"✅ Hasil disimpan:")
    print(f"   📁 {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    print(f"   📁 dilation_cross_{base_name}.jpg ({os.path.getsize(f'dilation_cross_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 dilation_ellipse_{base_name}.jpg ({os.path.getsize(f'dilation_ellipse_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 {plot_path} ({os.path.getsize(plot_path)/1024:.1f} KB)")
    
    # Show statistics
    print(f"\n📊 Dilation Statistics:")
    print(f"   • Original white pixels: {original_pixels:,}")
    print(f"   • Dilated white pixels: {dilated_pixels:,}")
    print(f"   • Pixels added: {dilated_pixels - original_pixels:,}")
    print(f"   • Increase: {(dilated_pixels - original_pixels)/original_pixels*100:.1f}%")
    
    print(f"\n🎯 Kegunaan Dilation:")
    print(f"   • Memperbesar objek")
    print(f"   • Mengisi lubang kecil")
    print(f"   • Menghubungkan objek yang terpisah dekat")
    print(f"   • Memperkuat struktur objek")

def main():
    """Main function"""
    print("🎯 Dilation - Morphological Operation")
    
    # Konfigurasi - ubah sesuai kebutuhan
    image_path = "/home/nugraha/Pictures/nugraha_10.jpeg"  # Ganti dengan path gambar Anda
    kernel_size = 5    # Ukuran kernel
    iterations = 1     # Jumlah iterasi
    
    # Apply dilation
    apply_dilation(image_path, kernel_size, iterations)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()