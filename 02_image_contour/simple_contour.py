#!/usr/bin/env python3
"""
Deteksi Kontur Eksternal - Versi yang Pasti Jalan
Soal CPMK 1.2: Kontur eksternal dengan 4-ketetanggaan dan 8-ketetanggaan
"""

import numpy as np

# Citra dari soal (1 = objek, 0 = background)
image = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
])

def print_matrix_fancy(matrix, title):
    """Print matrix dengan border yang bagus"""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Header kolom
    print("   ", end="")
    for j in range(matrix.shape[1]):
        print(f"{j:2}", end=" ")
    print()
    
    # Matrix dengan nomor baris
    for i, row in enumerate(matrix):
        print(f"{i:2}:", end=" ")
        for val in row:
            if val == 1:
                print(f"{'█':2}", end=" ")  # Block character untuk 1
            else:
                print(f"{'·':2}", end=" ")  # Dot untuk 0
        print(f"  <- Baris {i}")
    print()

def find_external_contour_4(image):
    """
    Mencari kontur eksternal dengan 4-ketetanggaan
    4-connectivity: atas, bawah, kiri, kanan
    """
    rows, cols = image.shape
    contour = np.zeros_like(image)
    
    # Tetangga 4-ketetanggaan: atas, bawah, kiri, kanan
    neighbors_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    print("Memproses 4-ketetanggaan...")
    contour_pixels = []
    
    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 1:  # Jika pixel adalah objek
                # Cek apakah ada tetangga yang background
                is_contour = False
                border_neighbors = []
                
                for di, dj in neighbors_4:
                    ni, nj = i + di, j + dj
                    
                    # Cek batas citra
                    if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                        is_contour = True
                        border_neighbors.append("batas")
                    # Cek tetangga background
                    elif image[ni, nj] == 0:
                        is_contour = True
                        border_neighbors.append(f"({ni},{nj})")
                
                if is_contour:
                    contour[i, j] = 1
                    contour_pixels.append((i, j, border_neighbors))
    
    print(f"Ditemukan {len(contour_pixels)} pixel kontur:")
    for i, j, neighbors in contour_pixels:
        print(f"  Pixel ({i},{j}) -> tetangga background: {neighbors}")
    
    return contour

def find_external_contour_8(image):
    """
    Mencari kontur eksternal dengan 8-ketetanggaan
    8-connectivity: semua arah termasuk diagonal
    """
    rows, cols = image.shape
    contour = np.zeros_like(image)
    
    # Tetangga 8-ketetanggaan: semua arah
    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                   (0, 1), (1, -1), (1, 0), (1, 1)]
    
    print("Memproses 8-ketetanggaan...")
    contour_pixels = []
    
    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 1:  # Jika pixel adalah objek
                # Cek apakah ada tetangga yang background
                is_contour = False
                border_neighbors = []
                
                for di, dj in neighbors_8:
                    ni, nj = i + di, j + dj
                    
                    # Cek batas citra
                    if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                        is_contour = True
                        border_neighbors.append("batas")
                    # Cek tetangga background
                    elif image[ni, nj] == 0:
                        is_contour = True
                        border_neighbors.append(f"({ni},{nj})")
                
                if is_contour:
                    contour[i, j] = 1
                    contour_pixels.append((i, j, border_neighbors))
    
    print(f"Ditemukan {len(contour_pixels)} pixel kontur:")
    for i, j, neighbors in contour_pixels:
        print(f"  Pixel ({i},{j}) -> tetangga background: {neighbors}")
    
    return contour

def compare_contours(contour_4, contour_8):
    """Bandingkan hasil kedua metode"""
    print("\n" + "="*60)
    print("PERBANDINGAN HASIL")
    print("="*60)
    
    # Statistik
    count_4 = np.sum(contour_4)
    count_8 = np.sum(contour_8)
    
    print(f"Jumlah pixel kontur (4-ketetanggaan): {count_4}")
    print(f"Jumlah pixel kontur (8-ketetanggaan): {count_8}")
    
    # Perbedaan
    diff = contour_4.astype(int) - contour_8.astype(int)
    unique_4 = np.where((contour_4 == 1) & (contour_8 == 0))
    unique_8 = np.where((contour_4 == 0) & (contour_8 == 1))
    
    if len(unique_4[0]) > 0:
        print(f"\nPixel yang hanya ada di 4-ketetanggaan:")
        for i, j in zip(unique_4[0], unique_4[1]):
            print(f"  Posisi ({i},{j})")
    
    if len(unique_8[0]) > 0:
        print(f"\nPixel yang hanya ada di 8-ketetanggaan:")
        for i, j in zip(unique_8[0], unique_8[1]):
            print(f"  Posisi ({i},{j})")
    
    if len(unique_4[0]) == 0 and len(unique_8[0]) == 0:
        print("\n✅ Hasil kedua metode SAMA PERSIS!")
    
    return diff

def save_results_to_file(image, contour_4, contour_8):
    """Simpan hasil ke file text"""
    filename = "contour_results.txt"
    
    with open(filename, 'w') as f:
        f.write("HASIL DETEKSI KONTUR EKSTERNAL\n")
        f.write("="*50 + "\n\n")
        
        f.write("CITRA ASLI:\n")
        for row in image:
            f.write(' '.join(map(str, row)) + '\n')
        
        f.write("\nKONTUR 4-KETETANGGAAN:\n")
        for row in contour_4:
            f.write(' '.join(map(str, row)) + '\n')
        
        f.write("\nKONTUR 8-KETETANGGAAN:\n")
        for row in contour_8:
            f.write(' '.join(map(str, row)) + '\n')
        
        f.write(f"\nSTATISTIK:\n")
        f.write(f"Pixel objek asli: {np.sum(image)}\n")
        f.write(f"Pixel kontur (4-conn): {np.sum(contour_4)}\n")
        f.write(f"Pixel kontur (8-conn): {np.sum(contour_8)}\n")
    
    print(f"\n💾 Hasil disimpan ke file: {filename}")

def main():
    """Fungsi utama"""
    print("🔍 DETEKSI KONTUR EKSTERNAL - SOAL CPMK 1.2")
    print("="*60)
    
    # Tampilkan citra asli
    print_matrix_fancy(image, "CITRA ASLI")
    print("Keterangan: █ = objek (1), · = background (0)")
    
    print(f"\nJumlah total pixel objek: {np.sum(image)}")
    
    # Proses kontur 4-ketetanggaan
    print("\n" + "="*60)
    print("PROSES 4-KETETANGGAAN")
    print("="*60)
    contour_4 = find_external_contour_4(image)
    
    # Proses kontur 8-ketetanggaan  
    print("\n" + "="*60)
    print("PROSES 8-KETETANGGAAN")
    print("="*60)
    contour_8 = find_external_contour_8(image)
    
    # Tampilkan hasil
    print("\n" + "="*60)
    print("HASIL AKHIR")
    print("="*60)
    
    print_matrix_fancy(contour_4, "KONTUR EKSTERNAL (4-KETETANGGAAN)")
    print_matrix_fancy(contour_8, "KONTUR EKSTERNAL (8-KETETANGGAAN)")
    
    # Bandingkan hasil
    compare_contours(contour_4, contour_8)
    
    # Simpan hasil
    save_results_to_file(image, contour_4, contour_8)
    
    # Penjelasan
    print("\n" + "="*60)
    print("PENJELASAN TEORI")
    print("="*60)
    print("🔹 4-Ketetanggaan:")
    print("   - Hanya mempertimbangkan tetangga horizontal & vertikal")
    print("   - Tetangga: atas, bawah, kiri, kanan (4 arah)")
    print("   - Kontur cenderung lebih tebal")
    
    print("\n🔹 8-Ketetanggaan:")
    print("   - Mempertimbangkan semua 8 tetangga termasuk diagonal")
    print("   - Tetangga: 4 arah + 4 diagonal")
    print("   - Kontur cenderung lebih tipis")
    
    print("\n🔹 Kontur Eksternal:")
    print("   - Pixel objek yang berbatasan dengan background")
    print("   - Atau pixel objek yang berada di tepi citra")
    
    print("\n✅ PROGRAM SELESAI!")
    print("="*60)

if __name__ == "__main__":
    main()