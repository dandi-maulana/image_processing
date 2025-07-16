#!/usr/bin/env python3
"""
Closing - Operasi Morfologi
Mengisi lubang dan celah kecil dalam objek
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
    print("🎨 Membuat gambar test untuk operasi Closing...")
    
    # Buat gambar binary dengan objek berlubang
    img = np.zeros((400, 600), dtype=np.uint8)
    
    # Objek dengan lubang kecil
    cv2.rectangle(img, (50, 50), (200, 150), 255, -1)      # Persegi
    cv2.rectangle(img, (80, 80), (120, 120), 0, -1)        # Lubang kecil
    cv2.rectangle(img, (150, 80), (170, 120), 0, -1)       # Lubang kecil lagi
    
    # Lingkaran dengan lubang
    cv2.circle(img, (300, 100), 60, 255, -1)               # Lingkaran besar
    cv2.circle(img, (280, 85), 8, 0, -1)                   # Lubang kecil
    cv2.circle(img, (320, 115), 6, 0, -1)                  # Lubang kecil
    cv2.circle(img, (290, 120), 7, 0, -1)                  # Lubang kecil
    
    # Objek dengan celah
    cv2.rectangle(img, (450, 50), (550, 150), 255, -1)     # Persegi
    cv2.rectangle(img, (490, 50), (510, 60), 0, -1)        # Celah atas
    cv2.rectangle(img, (490, 140), (510, 150), 0, -1)      # Celah bawah
    
    # Objek yang hampir terhubung
    cv2.rectangle(img, (50, 200), (150, 250), 255, -1)     # Kotak kiri
    cv2.rectangle(img, (160, 200), (260, 250), 255, -1)    # Kotak kanan
    # Celah kecil di tengah (akan ditutup oleh closing)
    
    # Huruf dengan celah
    cv2.rectangle(img, (300, 200), (320, 280), 255, -1)    # Vertikal kiri
    cv2.rectangle(img, (380, 200), (400, 280), 255, -1)    # Vertikal kanan
    cv2.rectangle(img, (320, 200), (380, 220), 255, -1)    # Horizontal atas
    cv2.rectangle(img, (320, 235), (380, 245), 255, -1)    # Horizontal tengah (putus)
    cv2.rectangle(img, (320, 260), (380, 280), 255, -1)    # Horizontal bawah
    # Buat celah kecil di horizontal tengah
    cv2.rectangle(img, (345, 235), (355, 245), 0, -1)
    
    # Objek dengan noise internal (lubang kecil)
    cv2.rectangle(img, (450, 200), (550, 280), 255, -1)
    # Tambahkan lubang kecil yang akan ditutup
    cv2.circle(img, (470, 220), 4, 0, -1)
    cv2.circle(img, (500, 240), 3, 0, -1)
    cv2.circle(img, (530, 260), 4, 0, -1)
    cv2.rectangle(img, (480, 250), (490, 260), 0, -1)
    cv2.rectangle(img, (510, 220), (520, 230), 0, -1)
    
    # Bentuk kompleks dengan banyak lubang kecil
    cv2.ellipse(img, (300, 350), (80, 30), 0, 0, 360, 255, -1)
    # Tambahkan banyak lubang kecil
    for i in range(250, 350, 15):
        for j in range(330, 370, 10):
            if np.random.random() > 0.7:  # 30% chance untuk lubang
                cv2.circle(img, (i, j), 2, 0, -1)
    
    cv2.imwrite("morphology_test.jpg", img)
    print("✅ Test image created: morphology_test.jpg")
    return "morphology_test.jpg"

def apply_closing(image_path, kernel_size=5, iterations=1):
    """
    Menerapkan operasi Closing pada gambar
    
    Args:
        image_path: Path ke gambar input
        kernel_size: Ukuran kernel (structuring element)
        iterations: Jumlah iterasi closing
    """
    print(f"🟡 CLOSING - Morphological Operation")
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
    
    # Apply closing (dilation followed by erosion)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # Manual closing for comparison
    dilated = cv2.dilate(img, kernel, iterations=iterations)
    manual_closed = cv2.erode(dilated, kernel, iterations=iterations)
    
    # Test with different kernel sizes
    kernels = [3, 5, 7, 9]
    closing_results = []
    
    for k_size in kernels:
        k = np.ones((k_size, k_size), np.uint8)
        closed_temp = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, iterations=1)
        closing_results.append(closed_temp)
    
    # Test with different kernel shapes
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    closed_cross = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_cross, iterations=iterations)
    closed_ellipse = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_ellipse, iterations=iterations)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save main result
    output_path = f"closing_{base_name}.jpg"
    cv2.imwrite(output_path, closed)
    
    # Save intermediate steps
    cv2.imwrite(f"closing_dilated_{base_name}.jpg", dilated)
    cv2.imwrite(f"closing_manual_{base_name}.jpg", manual_closed)
    cv2.imwrite(f"closing_cross_{base_name}.jpg", closed_cross)
    cv2.imwrite(f"closing_ellipse_{base_name}.jpg", closed_ellipse)
    
    # Create comparison plot
    plt.figure(figsize=(20, 15))
    
    # Row 1: Original and process steps
    plt.subplot(4, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(4, 4, 2)
    plt.imshow(dilated, cmap='gray')
    plt.title('Step 1: Dilation')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(closed, cmap='gray')
    plt.title('Step 2: Erosion (Closing)')
    plt.axis('off')
    
    # Show filled holes
    filled_holes = cv2.absdiff(closed, img)
    plt.subplot(4, 4, 4)
    plt.imshow(filled_holes, cmap='hot')
    plt.title('Filled Holes/Gaps')
    plt.axis('off')
    
    # Row 2: Different kernel sizes
    for i, (k_size, result) in enumerate(zip(kernels, closing_results)):
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
    plt.imshow(closed_cross, cmap='gray')
    plt.title('Closing with Cross')
    plt.axis('off')
    
    plt.subplot(4, 4, 11)
    plt.imshow(kernel_ellipse, cmap='gray')
    plt.title('Ellipse Kernel')
    plt.axis('off')
    
    plt.subplot(4, 4, 12)
    plt.imshow(closed_ellipse, cmap='gray')
    plt.title('Closing with Ellipse')
    plt.axis('off')
    
    # Row 4: Analysis
    plt.subplot(4, 4, 13)
    # Multiple iterations
    iter_results = []
    for iter_num in range(1, 6):
        iter_result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iter_num)
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
    closed_pixels = np.sum(closed == 255)
    pixel_counts = [original_pixels]
    
    for result in closing_results:
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
    
    plt.plot(range(1, 6), iter_pixel_counts, 'o-', color='orange', linewidth=2)
    plt.title('Pixels vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('White Pixels')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"closing_analysis_{base_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Show results
    print(f"✅ Hasil disimpan:")
    print(f"   📁 {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    print(f"   📁 closing_dilated_{base_name}.jpg ({os.path.getsize(f'closing_dilated_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 closing_manual_{base_name}.jpg ({os.path.getsize(f'closing_manual_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 {plot_path} ({os.path.getsize(plot_path)/1024:.1f} KB)")
    
    # Show statistics
    holes_filled = closed_pixels - original_pixels
    print(f"\n📊 Closing Statistics:")
    print(f"   • Original white pixels: {original_pixels:,}")
    print(f"   • Closed white pixels: {closed_pixels:,}")
    print(f"   • Holes filled: {holes_filled:,}")
    print(f"   • Fill ratio: {holes_filled/original_pixels*100:.1f}%")
    
    # Compare manual vs built-in closing
    manual_pixels = np.sum(manual_closed == 255)
    print(f"   • Manual closing pixels: {manual_pixels:,}")
    print(f"   • Built-in vs Manual difference: {abs(closed_pixels - manual_pixels)} pixels")
    
    print(f"\n🎯 Kegunaan Closing:")
    print(f"   • Mengisi lubang kecil dalam objek")
    print(f"   • Menghubungkan bagian yang terputus")
    print(f"   • Mengisi celah kecil")
    print(f"   • Smoothing kontur bagian dalam")
    print(f"   • Dilation diikuti Erosion")

def main():
    """Main function"""
    print("🎯 Closing - Morphological Operation")
    
    # Konfigurasi - ubah sesuai kebutuhan
    image_path = "/home/nugraha/Pictures/nugraha_07.jpeg"  # Ganti dengan path gambar Anda
    kernel_size = 5    # Ukuran kernel
    iterations = 1     # Jumlah iterasi
    
    # Apply closing
    apply_closing(image_path, kernel_size, iterations)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()