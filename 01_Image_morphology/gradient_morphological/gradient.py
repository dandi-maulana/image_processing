#!/usr/bin/env python3
"""
Morphological Gradient - Operasi Morfologi
Mendeteksi tepi objek dengan gradient morfologi
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
    print("🎨 Membuat gambar test untuk operasi Gradient...")
    
    # Buat gambar binary dengan berbagai bentuk
    img = np.zeros((400, 600), dtype=np.uint8)
    
    # Bentuk geometris untuk edge detection
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)      # Persegi
    cv2.circle(img, (250, 100), 50, 255, -1)               # Lingkaran
    cv2.ellipse(img, (400, 100), (60, 40), 0, 0, 360, 255, -1)  # Elips
    
    # Bentuk dengan ketebalan berbeda
    cv2.rectangle(img, (50, 200), (150, 250), 255, -1)     # Persegi tipis
    cv2.rectangle(img, (200, 180), (350, 270), 255, -1)    # Persegi tebal
    
    # Bentuk kompleks
    cv2.rectangle(img, (400, 180), (500, 270), 255, -1)    # Persegi
    cv2.rectangle(img, (420, 200), (480, 250), 0, -1)      # Lubang di tengah
    
    # Bentuk dengan noise
    cv2.rectangle(img, (50, 300), (200, 350), 255, -1)
    # Tambahkan beberapa lubang kecil
    cv2.circle(img, (80, 325), 5, 0, -1)
    cv2.circle(img, (120, 325), 4, 0, -1)
    cv2.circle(img, (160, 325), 6, 0, -1)
    
    # Bentuk dengan outline
    cv2.rectangle(img, (250, 300), (400, 350), 255, 3)     # Outline saja
    cv2.circle(img, (475, 325), 25, 255, 3)                # Circle outline
    
    # Teks untuk edge detection
    cv2.putText(img, 'GRADIENT', (420, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    
    cv2.imwrite("morphology_test.jpg", img)
    print("✅ Test image created: morphology_test.jpg")
    return "morphology_test.jpg"

def apply_gradient(image_path, kernel_size=5):
    """
    Menerapkan operasi Morphological Gradient pada gambar
    
    Args:
        image_path: Path ke gambar input
        kernel_size: Ukuran kernel (structuring element)
    """
    print(f"🟠 MORPHOLOGICAL GRADIENT - Edge Detection")
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
    print(f"🔧 Kernel size: {kernel_size}x{kernel_size}")
    
    # Apply morphological gradient
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    
    # Manual gradient calculation
    dilated = cv2.dilate(img, kernel, iterations=1)
    eroded = cv2.erode(img, kernel, iterations=1)
    manual_gradient = cv2.subtract(dilated, eroded)
    
    # External gradient (dilation - original)
    external_gradient = cv2.subtract(dilated, img)
    
    # Internal gradient (original - erosion)
    internal_gradient = cv2.subtract(img, eroded)
    
    # Test with different kernel sizes
    kernels = [3, 5, 7, 9]
    gradient_results = []
    
    for k_size in kernels:
        k = np.ones((k_size, k_size), np.uint8)
        grad_temp = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)
        gradient_results.append(grad_temp)
    
    # Test with different kernel shapes
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    gradient_cross = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel_cross)
    gradient_ellipse = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel_ellipse)
    
    # Compare with traditional edge detection
    # Sobel edge detection
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
    
    # Canny edge detection
    canny = cv2.Canny(img, 50, 150)
    
    # Save results
    base_name = Path(image_path).stem
    
    # Save main result
    output_path = f"gradient_{base_name}.jpg"
    cv2.imwrite(output_path, gradient)
    
    # Save different types
    cv2.imwrite(f"gradient_external_{base_name}.jpg", external_gradient)
    cv2.imwrite(f"gradient_internal_{base_name}.jpg", internal_gradient)
    cv2.imwrite(f"gradient_manual_{base_name}.jpg", manual_gradient)
    cv2.imwrite(f"gradient_cross_{base_name}.jpg", gradient_cross)
    cv2.imwrite(f"gradient_ellipse_{base_name}.jpg", gradient_ellipse)
    
    # Create comparison plot
    plt.figure(figsize=(20, 16))
    
    # Row 1: Original and gradient types
    plt.subplot(4, 5, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(4, 5, 2)
    plt.imshow(dilated, cmap='gray')
    plt.title('Dilation')
    plt.axis('off')
    
    plt.subplot(4, 5, 3)
    plt.imshow(eroded, cmap='gray')
    plt.title('Erosion')
    plt.axis('off')
    
    plt.subplot(4, 5, 4)
    plt.imshow(gradient, cmap='gray')
    plt.title('Morphological Gradient\n(Dilation - Erosion)')
    plt.axis('off')
    
    plt.subplot(4, 5, 5)
    plt.imshow(manual_gradient, cmap='gray')
    plt.title('Manual Gradient\n(Verification)')
    plt.axis('off')
    
    # Row 2: Different gradient types
    plt.subplot(4, 5, 6)
    plt.imshow(external_gradient, cmap='gray')
    plt.title('External Gradient\n(Dilation - Original)')
    plt.axis('off')
    
    plt.subplot(4, 5, 7)
    plt.imshow(internal_gradient, cmap='gray')
    plt.title('Internal Gradient\n(Original - Erosion)')
    plt.axis('off')
    
    plt.subplot(4, 5, 8)
    plt.imshow(gradient_cross, cmap='gray')
    plt.title('Gradient with Cross Kernel')
    plt.axis('off')
    
    plt.subplot(4, 5, 9)
    plt.imshow(gradient_ellipse, cmap='gray')
    plt.title('Gradient with Ellipse Kernel')
    plt.axis('off')
    
    plt.subplot(4, 5, 10)
    # Show kernel shapes
    fig_kernel = plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(kernel, cmap='gray')
    plt.title('Rectangle Kernel')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(kernel_cross, cmap='gray')
    plt.title('Cross Kernel')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(kernel_ellipse, cmap='gray')
    plt.title('Ellipse Kernel')
    plt.axis('off')
    
    plt.tight_layout()
    kernel_path = f"gradient_kernels_{base_name}.png"
    plt.savefig(kernel_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Continue with main plot
    plt.figure(figsize=(20, 16))
    
    # Row 1: Original and basic gradients
    plt.subplot(4, 5, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(4, 5, 2)
    plt.imshow(gradient, cmap='gray')
    plt.title('Morphological Gradient')
    plt.axis('off')
    
    plt.subplot(4, 5, 3)
    plt.imshow(external_gradient, cmap='gray')
    plt.title('External Gradient')
    plt.axis('off')
    
    plt.subplot(4, 5, 4)
    plt.imshow(internal_gradient, cmap='gray')
    plt.title('Internal Gradient')
    plt.axis('off')
    
    plt.subplot(4, 5, 5)
    # Combined gradients
    combined_gradient = cv2.addWeighted(external_gradient, 0.5, internal_gradient, 0.5, 0)
    plt.imshow(combined_gradient, cmap='gray')
    plt.title('Combined Gradient')
    plt.axis('off')
    
    # Row 2: Different kernel sizes
    for i, (k_size, result) in enumerate(zip(kernels, gradient_results)):
        plt.subplot(4, 5, 6 + i)
        plt.imshow(result, cmap='gray')
        plt.title(f'Kernel {k_size}x{k_size}')
        plt.axis('off')
    
    plt.subplot(4, 5, 10)
    plt.imshow(gradient_cross, cmap='gray')
    plt.title('Cross Kernel')
    plt.axis('off')
    
    # Row 3: Comparison with traditional edge detection
    plt.subplot(4, 5, 11)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')
    
    plt.subplot(4, 5, 12)
    plt.imshow(canny, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    
    plt.subplot(4, 5, 13)
    plt.imshow(gradient, cmap='gray')
    plt.title('Morphological Gradient')
    plt.axis('off')
    
    # Edge thickness comparison
    plt.subplot(4, 5, 14)
    # Create thick edges with larger kernel
    thick_kernel = np.ones((9, 9), np.uint8)
    thick_gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, thick_kernel)
    plt.imshow(thick_gradient, cmap='gray')
    plt.title('Thick Edges (9x9 kernel)')
    plt.axis('off')
    
    plt.subplot(4, 5, 15)
    # Create thin edges with smaller kernel
    thin_kernel = np.ones((3, 3), np.uint8)
    thin_gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, thin_kernel)
    plt.imshow(thin_gradient, cmap='gray')
    plt.title('Thin Edges (3x3 kernel)')
    plt.axis('off')
    
    # Row 4: Analysis
    plt.subplot(4, 5, 16)
    # Edge pixel count analysis
    edge_counts = []
    for result in gradient_results:
        edge_counts.append(np.sum(result > 0))
    
    plt.bar(range(len(edge_counts)), edge_counts, color=['red', 'green', 'blue', 'orange'])
    plt.title('Edge Pixels Count')
    plt.xlabel('Kernel Size')
    plt.ylabel('Edge Pixels')
    plt.xticks(range(len(edge_counts)), [f'{k}x{k}' for k in kernels])
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 5, 17)
    # Gradient intensity histogram
    plt.hist(gradient.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.title('Gradient Intensity Distribution')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 5, 18)
    # Line profile analysis
    center_row = img.shape[0] // 2
    plt.plot(img[center_row, :], label='Original', linewidth=2)
    plt.plot(gradient[center_row, :], label='Gradient', linewidth=2)
    plt.title('Horizontal Line Profile')
    plt.xlabel('X Position')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 5, 19)
    # Comparison of different methods
    methods = ['Morphological', 'Sobel', 'Canny']
    method_counts = [
        np.sum(gradient > 0),
        np.sum(sobel_combined > 0),
        np.sum(canny > 0)
    ]
    
    plt.bar(methods, method_counts, color=['orange', 'blue', 'green'])
    plt.title('Edge Detection Comparison')
    plt.ylabel('Edge Pixels')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 5, 20)
    # Gradient vs different shapes
    shape_results = [
        np.sum(gradient > 0),
        np.sum(gradient_cross > 0),
        np.sum(gradient_ellipse > 0)
    ]
    shape_names = ['Rectangle', 'Cross', 'Ellipse']
    
    plt.bar(shape_names, shape_results, color=['red', 'green', 'blue'])
    plt.title('Kernel Shape Comparison')
    plt.ylabel('Edge Pixels')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"gradient_analysis_{base_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Show results
    print(f"✅ Hasil disimpan:")
    print(f"   📁 {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    print(f"   📁 gradient_external_{base_name}.jpg ({os.path.getsize(f'gradient_external_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 gradient_internal_{base_name}.jpg ({os.path.getsize(f'gradient_internal_{base_name}.jpg')/1024:.1f} KB)")
    print(f"   📁 {plot_path} ({os.path.getsize(plot_path)/1024:.1f} KB)")
    print(f"   📁 {kernel_path} ({os.path.getsize(kernel_path)/1024:.1f} KB)")
    
    # Show statistics
    original_pixels = np.sum(img == 255)
    gradient_pixels = np.sum(gradient > 0)
    external_pixels = np.sum(external_gradient > 0)
    internal_pixels = np.sum(internal_gradient > 0)
    
    print(f"\n📊 Gradient Statistics:")
    print(f"   • Original white pixels: {original_pixels:,}")
    print(f"   • Gradient edge pixels: {gradient_pixels:,}")
    print(f"   • External gradient pixels: {external_pixels:,}")
    print(f"   • Internal gradient pixels: {internal_pixels:,}")
    print(f"   • Edge ratio: {gradient_pixels/original_pixels*100:.1f}%")
    
    # Compare different methods
    sobel_pixels = np.sum(sobel_combined > 0)
    canny_pixels = np.sum(canny > 0)
    
    print(f"\n📈 Edge Detection Comparison:")
    print(f"   • Morphological gradient: {gradient_pixels:,} pixels")
    print(f"   • Sobel edge detection: {sobel_pixels:,} pixels")
    print(f"   • Canny edge detection: {canny_pixels:,} pixels")
    
    print(f"\n🎯 Kegunaan Morphological Gradient:")
    print(f"   • Mendeteksi tepi objek")
    print(f"   • Menghasilkan tepi yang tebal dan jelas")
    print(f"   • Cocok untuk objek binary")
    print(f"   • Dapat dikontrol dengan kernel shape")
    print(f"   • Dilation - Erosion")

def main():
    """Main function"""
    print("🎯 Morphological Gradient - Edge Detection")
    
    # Konfigurasi - ubah sesuai kebutuhan
    image_path = "/home/nugraha/Pictures/nugraha_07.jpeg"  # Ganti dengan path gambar Anda
    kernel_size = 5    # Ukuran kernel
    
    # Apply gradient
    apply_gradient(image_path, kernel_size)
    
    print(f"\n🎉 Selesai! Lihat hasil dengan: eog *.jpg *.png")

if __name__ == "__main__":
    main()