# 🖼️ Praktikum Pengolahan Citra Digital - Pertemuan 2

Aplikasi interaktif Python untuk mempelajari dan mempraktikkan:
1. **Operasi Piksel (Aras Titik / Point Operations)**
2. **Operasi Aritmatika Citra (Arithmetic Operations)**
3. **Operasi Logika (Boolean / Bitwise Operations)**
4. **Inspeksi Matriks Piksel (Pixel & Neighborhood Matrix Inspector)**

---

## 🚀 Cara Menjalankan Aplikasi

Tersedia **2 Pilihan Antarmuka**:

### 1. Antarmuka Desktop GUI (Tkinter + OpenCV)
Bisa langsung mengklik piksel pada citra untuk melihat matriks tetangga $3\times3$ / $5\times5$ secara instan:
```bash
python3 app_gui.py
```

### 2. Antarmuka Web (Streamlit)
Visualisasi berbasis browser interaktif dengan plot matriks & heatmap:
```bash
python3 -m streamlit run app_streamlit.py
```
*(atau `streamlit run app_streamlit.py` jika path streamlit sudah ada di environment PATH)*

---

## 📚 Ringkasan Materi & Rumus Matematika

### 1. Operasi Piksel (Aras Titik)
Operasi yang dilakukan pada setiap piksel secara individual tanpa terpengaruh oleh piksel tetangganya:
* **Pencerahan (Brightness):**
  $$O(x,y) = \text{clamp}(I(x,y) + b, 0, 255)$$
* **Peregangan Kontras (Contrast):**
  $$O(x,y) = \text{clamp}(128 + c \cdot (I(x,y) - 128), 0, 255)$$
* **Citra Negatif (Invert):**
  $$O(x,y) = 255 - I(x,y)$$
* **Pengambangan (Thresholding):**
  $$O(x,y) = \begin{cases} 255, & \text{jika } I(x,y) \ge T \\ 0, & \text{jika } I(x,y) < T \end{cases}$$
* **Koreksi Gamma (Power-Law):**
  $$O(x,y) = 255 \cdot \left(\frac{I(x,y)}{255}\right)^{1/\gamma}$$
* **Transformasi Logaritmik:**
  $$O(x,y) = c \cdot \ln(1 + I(x,y)), \quad c = \frac{255}{\ln(1 + \max(I))}$$
* **Grayscale Luminance:**
  $$Y = 0.299R + 0.587G + 0.114B$$

---

### 2. Operasi Aritmatika Citra
Operasi matematika antara dua citra ($A$ dan $B$) atau antara citra dengan nilai skalar ($c$):
* **Penjumlahan (Addition):**
  * Mode Saturasi (Clamping): $O(x,y) = \min(A(x,y) + B(x,y), 255)$
  * Mode Modulo: $O(x,y) = (A(x,y) + B(x,y)) \pmod{256}$
* **Pengurangan (Subtraction):**
  * Standar: $O(x,y) = \max(A(x,y) - B(x,y), 0)$
  * Selisih Mutlak (Difference): $O(x,y) = |A(x,y) - B(x,y)|$
* **Perkalian (Multiplication):**
  $$O(x,y) = \frac{A(x,y) \cdot B(x,y)}{255}$$
* **Pembagian (Division):**
  $$O(x,y) = \frac{A(x,y)}{B(x,y) + 1} \cdot 255$$
* **Pencampuran Citra Berbobot (Blending):**
  $$O(x,y) = \alpha \cdot A(x,y) + (1 - \alpha) \cdot B(x,y), \quad \alpha \in [0, 1]$$

---

### 3. Operasi Logika (Boolean / Bitwise)
Operasi biner bit demi bit (8-bit) pada citra dengan citra lain atau masker geometri:
* **Bitwise AND ($\land$):** Digunakan untuk operasi *masking* (memotong objek sesuai bentuk masker).
* **Bitwise OR ($\lor$):** Digunakan untuk menggabungkan citra dan latar belakang.
* **Bitwise NOT ($\neg$):** Membalikkan nilai bit ($255 - A$).
* **Bitwise XOR ($\oplus$):** Menyorot perbedaan atau batas outline antar dua citra.
* **Bitwise NAND & NOR:** Kombinasi NOT dengan AND / OR.

---

### 4. Fitur Inspeksi Nilai Matriks Piksel
* **Koordinat $(X, Y)$**: Posisi piksel pada citra (kolom $X$, baris $Y$).
* **Format Nilai**:
  * RGB: `[R, G, B]` (rentang 0–255)
  * HEX: `#RRGGBB`
  * Intensitas Grayscale: $Y = 0.299R + 0.587G + 0.114B$
  * Biner 8-bit: misal `11010100`
* **Matriks Tetangga $3\times3$ atau $5\times5$**:
  Menampilkan tabel matriks nilai intensitas piksel tetangga di sekitar koordinat yang dipilih dengan penanda khusus pada titik pusat.
* **Kalkulasi Matematis Titik**:
  Menampilkan proses perhitungan nilai sebelum dan sesudah operasi diterapkan.

---

## 📁 Struktur Direktori
```
Pertemuan 2/
├── app_gui.py           # Aplikasi Desktop GUI (Tkinter + PIL + OpenCV)
├── app_streamlit.py     # Aplikasi Web Interaktif (Streamlit)
├── image_operations.py  # Modul inti rumus operasi citra & inspeksi matriks
├── generate_samples.py  # Script pembuat citra contoh (sample A & B)
├── samples/             # Direktori gambar sampel pengujian
└── README.md            # Dokumentasi & panduan
```
