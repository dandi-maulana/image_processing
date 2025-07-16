#!/usr/bin/env python3
"""
Top Hat - Operasi Morfologi
Mendeteksi objek kecil yang lebih terang dari background
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
    print("🎨 Membuat gambar test untuk operasi Top Hat...")
    
    # Buat gambar grayscale dengan objek kecil terang
    img = np.full((400, 600), 100, dtype=np.uint8)  # Background abu-abu
    
    # Tambahkan objek besar (akan dihilangkan oleh opening)
    cv2.rectangle(img, (50, 50), (200, 150), 180, -1)      # Objek besar
    cv2.circle(img, (350, 100), 60, 170, -1)               # Lingkaran besar
    
    # Tambahkan objek kecil terang (akan dideteksi oleh Top Hat)
    cv2.circle(img, (100, 200), 8, 255, -1)    # Titik terang kecil
    cv2.circle(img, (150, 220), 6, 255, -1)    # Titik terang kecil
    cv2.circle(img, (200, 200), 7, 255, -1)    # Titik terang kecil
    cv2.circle(img, (250, 210), 5, 255, -1)    # Titik terang kecil
    
    # Tambahkan garis tipis terang
    cv2.line(img, (300, 200), (400, 220), 255, 2)
    cv2.line(img, (420, 200), (520, 210), 255, 3)
    
    # Tambahkan noise berupa titik-titik kecil
    noise_points = [(80, 280), (120, 290), (160, 285), (200, 295), 
                   (240, 280), (280, 290), (320, 285), (360, 295),
                   (400, 280), (440, 290), (480, 285), (520, 295)]
    
    for point in noise_points:
        cv2.circle(img, point, 3, 255, -1)
    
    # Tambahkan objek kecil dalam bentuk persegi
    cv2.rectangle(img, (100, 320), (110, 330), 255, -1)
    cv2.rectangle(img, (150, 320), (160, 330), 255, -1)
    cv2.rectangle(img, (200, 320), (210, 330), 255, -1)
    cv2.rectangle(img, (250, 320), (260, 330), 255, -1)
    
    # Tambahkan teks kecil (akan dideteksi sebagai objek kecil)
    cv2.putText(img, 'small', (400, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    cv2.putText(img, 'text', (450, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    
    # Tambahkan variasi background
    for i in range(0, 600, 50):
        cv2.rectangle(img, (i, 350), (i+25, 380), 90, -1)   # Area lebih gelap
        cv2.rectangle(img, (i+25, 350), (i+50, 380), 110, -1)  # Area lebih terang
    
    # Tambahkan objek kecil di area dengan background berbeda
    cv2.circle(img, (75, 365), 4, 255, -1)
    cv2.circle(img, (125, 365), 4, 255, -1)
    cv2.circle(img, (175, 365), 4, 255, -1)
    cv2.circle(img, (225, 365), 4, 255, -1)
    
    cv2.imwrite("morphology_test.jpg", img)
    print("✅ Test image created: morphology_test.jpg")
    return "morphology_test.jpg"

def apply_tophat(image_path, kernel_size=15):
    """
    Menerapkan operasi Top Hat pada gambar
    
    Args:
        image_path: Path ke gambar input
        kernel_size: Ukuran kernel (structuring element)
    """
    print(f"🎩 TOP HAT - Morphological Operation")
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
    
    # Create structuring element (kernel)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    print(f"🔧 Kernel size: {kernel_size}x{kernel_size}")
    
    # Apply top hat (original - opening)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    
    # Manual top hat calculation
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    manual_tophat = cv2.subtract(img, opened)
    
    # Black hat (closing - original)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    manual_blackhat = cv2.subtract(closed, img)
    
    # Test with different kernel sizes
    kernels = [5, 10, 15, 20]
    tophat_results = []
    
    for k_size in kernels:
        k = np.ones((k_size, k_size), np.uint8)
        tophat_temp = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k)
        tophat_results.append(tophat_temp)
    
    # Test with different kernel shapes
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    tophat_cross = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_cross)
    tophat_ellipse = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_ellipse)
    
    # Enhanced top hat with thresholding
    _, tophat_thresh = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save main result
    output_path = f"tophat_{base_name}.jpg"
    cv2.imwrite(output_path, tophat)
    
    # Save other results
    cv2.imwrite(f"tophat_opened_{base_name}.jpg", opened)
    cv2.imwrite(f"tophat_manual_{base_name}.jpg", manual_tophat)
    cv2.imwrite(f"tophat_blackhat_{base_name}.jpg", blackhat)
    cv2.imwrite(f"tophat_thresh_{base_name}.jpg", tophat_thresh)
    cv2.imwrite(f"tophat_cross_{base_name}.jpg", tophat_cross)
    cv2.imwrite(f"tophat_ellipse_{base_name}.jpg", tophat_ellipse)
    
    # Create comparison plot
    plt.figure(figsize=(20, 16))
    
    # Row 1: Original and basic operations
    plt.subplot(4, 5, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(4, 5, 2)
    plt.imshow(opened, cmap='gray')
    plt.title('Opening')
    plt.axis('off')
    
    plt.subplot(4, 5, 3)
    plt.imshow(tophat, cmap='gray')
    plt.title('Top Hat\n(Original - Opening)')
    plt.axis('off')
    
    plt.subplot(4, 5, 4)
    plt.imshow(manual_tophat, cmap='gray')
    plt.title('Manual Top Hat\n(Verification)')
    plt.axis('off')
    
    plt.subplot(4, 5, 5)
    plt.imshow(tophat_thresh, cmap='gray')
    plt.title('Top Hat Thresholded')
    plt.axis('off')
    
    # Row 2: Different kernel sizes
    for i, (k_size, result) in enumerate(zip(kernels, tophat_results)):
        plt.subplot(4, 5, 6 + i)
        plt.imshow(result, cmap='gray')
        plt.title(f'Kernel {k_size}x{k_size}')
        plt.axis('off')
    
    plt.subplot(4, 5, 10)
    plt.imshow(tophat_cross, cmap='gray')
    plt.title('Cross Kernel')
    plt.axis('off')
    
    # Row 3: Comparison with Black Hat
    plt.subplot(4, 5, 11)
    plt.imshow(blackhat, cmap='gray')
    plt.title('Black Hat\n(Closing - Original)')
    plt.axis('off')
    
    plt.subplot(4, 5, 12)
    plt.imshow(closed, cmap='gray')
    plt.title('Closing')
    plt.axis('off')
    
    plt.subplot(4, 5, 13)
    plt.imshow(manual_blackhat, cmap='gray')
    plt.title('Manual Black Hat')
    plt.axis('off')
    
    plt.subplot(4, 5, 14)
    plt.imshow(tophat_ellipse, cmap='gray')
    plt.title('Ellipse Kernel')
    plt.axis('off')
    
    # Combined top hat and black hat
    plt.subplot(4, 5, 15)
    combined = cv2.addWeighted(tophat, 0.5, blackhat, 0.5, 0)
    plt.imshow(combined, cmap='gray')
    plt.title('Combined Top+Black Hat')
    plt.axis('off')
    
    # Row 4: Analysis
    plt.subplot(4, 5, 16)
    # Intensity histograms
    plt.hist(img.flatten(), bins=50, alpha=0.5, label='Original', color='blue')
    plt.hist(tophat.flatten(), bins=50, alpha=0.5, label='Top Hat', color='red')
    plt.title('Intensity Histograms')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 5, 17)
    # Detection statistics
    detected_pixels = []
    for result in tophat_results:
        detected_pixels.append(np.sum(result > 30))  # Threshold untuk deteksi
    
    plt.bar(range(len(detected_pixels)), detected_pixels, color=['red', 'green', 'blue', 'orange'])
    plt.title('Detected Small Objects')
    plt.xlabel('Kernel Size')
    plt.ylabel('Detected Pixels')
    plt.xticks(range(len(detected_pixels)), [f'{k}x{k}' for k in kernels])
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 5, 18)
    # Line profile analysis
    center_row = img.shape[0] // 2
    plt.plot(img[center_row, :], label='Original', linewidth=2)
    plt.plot(opened[center_row, :], label='Opening', linewidth=2)
    plt.plot(tophat[center_row, :], label='Top Hat', linewidth=2)
    plt.title('Horizontal Line Profile')
    plt.xlabel('X Position')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 5, 19)
    # Kernel size effect
    kernel_effects = []
    for k_size in kernels:
        k = np.ones((k_size, k_size), np.uint8)
        th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k)
        kernel_effects.append(np.mean(th))
    
    plt.plot(kernels, kernel_effects, 'o-', linewidth=2, markersize=8)
    plt.title('Kernel Size Effect')
    plt.xlabel('Kernel Size')
    plt.ylabel('Mean Top Hat Intensity')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 5, 20)
    # Detection comparison
    tophat_detection = np.sum(tophat > 30)
    blackhat_detection = np.sum(blackhat > 30)
    
    plt.bar(['Top Hat', 'Black Hat'], [tophat_detection, blackhat_detection], 
            color=['white', 'black'], edgecolor='gray', linewidth=2)
    plt.title('Detection Comparison')
    plt.ylabel('Detected Pixels')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"tophat_analysis_{base_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Show results
    print(f"✅ Hasil disimpan:")
    print(f"   📁 {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    print(f"   📁 tophat_opened_{base_name}.jpg ({os.path.getsize(f'tophat_opened_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 tophat_blackhat_{base_name}.jpg ({os.path.getsize(f'tophat_blackhat_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 tophat_thresh_{base_name}.jpg ({os.path.getsize(f'tophat_thresh_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 {plot_path} ({os.path.getsize(plot_path)/1024:.1f} KB)")
    
    # Show statistics
    original_mean = np.mean(img)
    tophat_mean = np.mean(tophat)
    blackhat_mean = np.mean(blackhat)
    
    tophat_max = np.max(tophat)
    detected_objects = np.sum(tophat > 30)
    
    print(f"\n📊 Top Hat Statistics:")
    print(f"   • Original image mean: {original_mean:.1f}")
    print(f"   • Top Hat mean: {tophat_mean:.1f}")
    print(f"   • Black Hat mean: {blackhat_mean:.1f}")
    print(f"   • Top Hat max value: {tophat_max}")
    print(f"   • Detected small objects: {detected_objects:,} pixels")
    print(f"   • Detection ratio: {detected_objects/img.size*100:.3f}%")
    
    # Compare manual vs built-in
    manual_pixels = np.sum(manual_tophat > 30)
    builtin_pixels = np.sum(tophat > 30)
    
    print(f"\n🔍 Verification:")
    print(f"   • Manual Top Hat detection: {manual_pixels:,} pixels")
    print(f"   • Built-in Top Hat detection: {builtin_pixels:,} pixels")
    print(f"   • Difference: {abs(manual_pixels - builtin_pixels)} pixels")
    
    print(f"\n🎯 Kegunaan Top Hat:")
    print(f"   • Mendeteksi objek kecil terang")
    print(f"   • Menghilangkan background yang tidak rata")
    print(f"   • Isolasi detail kecil")
    print(f"   • Deteksi noise atau partikel")
    print(f"   • Original - Opening")
    
    print(f"\n🎯 Kegunaan Black Hat:")
    print(f"   • Mendeteksi objek kecil gelap")
    print(f"   • Mendeteksi lubang atau celah")
    print(f"   • Closing - Original")

def main():
    """Main function"""
    print("🎯 Top Hat - Morphological Operation")
    
    # Konfigurasi - ubah sesuai kebutuhan
    image_path = "/home/nugraha/Pictures/nugraha_06.jpeg"  # Ganti dengan path gambar Anda
    kernel_size = 15   # Ukuran kernel (harus lebih besar dari objek yang ingin dideteksi)
    
    # Apply top hat
    apply_tophat(image_path, kernel_size)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()