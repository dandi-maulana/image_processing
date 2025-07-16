#!/usr/bin/env python3
"""
Erosion - Operasi Morfologi
Mengecilkan objek dan menghilangkan noise kecil
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
    print("🎨 Membuat gambar test untuk operasi Erosion...")
    
    # Buat gambar binary dengan berbagai bentuk
    img = np.zeros((400, 600), dtype=np.uint8)
    
    # Tambahkan berbagai bentuk geometris
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)      # Persegi
    cv2.circle(img, (250, 100), 50, 255, -1)               # Lingkaran
    cv2.ellipse(img, (400, 100), (60, 40), 0, 0, 360, 255, -1)  # Elips
    
    # Tambahkan bentuk yang terhubung tipis
    cv2.rectangle(img, (50, 200), (150, 250), 255, -1)     # Kotak kiri
    cv2.rectangle(img, (200, 200), (300, 250), 255, -1)    # Kotak kanan
    cv2.rectangle(img, (150, 220), (200, 230), 255, -1)    # Penghubung tipis
    
    # Tambahkan noise kecil
    cv2.circle(img, (450, 200), 3, 255, -1)
    cv2.circle(img, (500, 220), 2, 255, -1)
    cv2.circle(img, (520, 200), 4, 255, -1)
    
    # Tambahkan garis tipis
    cv2.line(img, (50, 300), (550, 300), 255, 2)
    cv2.line(img, (50, 320), (550, 320), 255, 4)
    cv2.line(img, (50, 340), (550, 340), 255, 6)
    
    # Tambahkan teks
    cv2.putText(img, 'EROSION TEST', (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    cv2.imwrite("morphology_test.jpg", img)
    print("✅ Test image created: morphology_test.jpg")
    return "morphology_test.jpg"

def apply_erosion(image_path, kernel_size=5, iterations=1):
    """
    Menerapkan operasi Erosion pada gambar
    
    Args:
        image_path: Path ke gambar input
        kernel_size: Ukuran kernel (structuring element)
        iterations: Jumlah iterasi erosion
    """
    print(f"🔴 EROSION - Morphological Operation")
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
    
    # Apply erosion
    eroded = cv2.erode(img, kernel, iterations=iterations)
    
    # Test with different kernel sizes
    kernels = [3, 5, 7, 9]
    erosion_results = []
    
    for k_size in kernels:
        k = np.ones((k_size, k_size), np.uint8)
        eroded_temp = cv2.erode(img, k, iterations=1)
        erosion_results.append(eroded_temp)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save main result
    output_path = f"erosion_{base_name}.jpg"
    cv2.imwrite(output_path, eroded)
    
    # Create comparison plot
    plt.figure(figsize=(20, 12))
    
    # Row 1: Original and main result
    plt.subplot(3, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(eroded, cmap='gray')
    plt.title(f'Erosion (kernel={kernel_size}x{kernel_size})')
    plt.axis('off')
    
    # Show kernel
    plt.subplot(3, 4, 3)
    plt.imshow(kernel, cmap='gray')
    plt.title(f'Structuring Element\n({kernel_size}x{kernel_size})')
    plt.axis('off')
    
    # Show difference
    difference = cv2.absdiff(img, eroded)
    plt.subplot(3, 4, 4)
    plt.imshow(difference, cmap='hot')
    plt.title('Removed Pixels (Difference)')
    plt.axis('off')
    
    # Row 2: Different kernel sizes
    for i, (k_size, result) in enumerate(zip(kernels, erosion_results)):
        plt.subplot(3, 4, 5 + i)
        plt.imshow(result, cmap='gray')
        plt.title(f'Kernel {k_size}x{k_size}')
        plt.axis('off')
    
    # Row 3: Analysis
    plt.subplot(3, 4, 9)
    # Multiple iterations
    iter_results = []
    for iter_num in range(1, 6):
        iter_result = cv2.erode(img, kernel, iterations=iter_num)
        iter_results.append(iter_result)
    
    # Show iteration 3 as example
    plt.imshow(iter_results[2], cmap='gray')
    plt.title('3 Iterations')
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.imshow(iter_results[4], cmap='gray')
    plt.title('5 Iterations')
    plt.axis('off')
    
    # Statistical analysis
    plt.subplot(3, 4, 11)
    original_pixels = np.sum(img == 255)
    eroded_pixels = np.sum(eroded == 255)
    pixel_counts = [original_pixels]
    
    for result in erosion_results:
        pixel_counts.append(np.sum(result == 255))
    
    plt.bar(range(len(pixel_counts)), pixel_counts, color=['blue', 'red', 'green', 'orange', 'purple'])
    plt.title('White Pixels Count')
    plt.xlabel('Kernel Size')
    plt.ylabel('Pixel Count')
    plt.xticks(range(len(pixel_counts)), ['Original'] + [f'{k}x{k}' for k in kernels])
    plt.grid(True, alpha=0.3)
    
    # Iteration analysis
    plt.subplot(3, 4, 12)
    iter_pixel_counts = []
    for result in iter_results:
        iter_pixel_counts.append(np.sum(result == 255))
    
    plt.plot(range(1, 6), iter_pixel_counts, 'o-', color='red', linewidth=2)
    plt.title('Pixels vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('White Pixels')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"erosion_analysis_{base_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Show results
    print(f"✅ Hasil disimpan:")
    print(f"   📁 {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    print(f"   📁 {plot_path} ({os.path.getsize(plot_path)/1024:.1f} KB)")
    
    # Show statistics
    print(f"\n📊 Erosion Statistics:")
    print(f"   • Original white pixels: {original_pixels:,}")
    print(f"   • Eroded white pixels: {eroded_pixels:,}")
    print(f"   • Pixels removed: {original_pixels - eroded_pixels:,}")
    print(f"   • Reduction: {(original_pixels - eroded_pixels)/original_pixels*100:.1f}%")
    
    print(f"\n🎯 Kegunaan Erosion:")
    print(f"   • Mengecilkan objek")
    print(f"   • Menghilangkan noise kecil")
    print(f"   • Memisahkan objek yang terhubung tipis")
    print(f"   • Menghilangkan detail kecil")

def main():
    """Main function"""
    print("🎯 Erosion - Morphological Operation")
    
    # Konfigurasi - ubah sesuai kebutuhan
    image_path = "/home/nugraha/Pictures/nugraha_01.jpeg"  # Ganti dengan path gambar Anda
    kernel_size = 5    # Ukuran kernel
    iterations = 1     # Jumlah iterasi
    
    # Apply erosion
    apply_erosion(image_path, kernel_size, iterations)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()