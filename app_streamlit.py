"""
app_streamlit.py
Aplikasi Web Interaktif Pengolahan Citra Digital - Praktikum Pertemuan 2
1. Operasi Piksel (Aras Titik)
2. Operasi Aritmatika Citra
3. Operasi Logika (Boolean) Citra
+ Inspeksi Nilai Matriks Piksel Interaktif
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

import image_operations as ops

st.set_page_config(
    page_title="PCD Pertemuan 2 - Operasi Citra Digital",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 18px;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .stSlider {
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🖼️ Pengolahan Citra Digital - Praktikum Pertemuan 2</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Materi: Operasi Piksel (Aras Titik) | Operasi Aritmatika Citra | Operasi Logika (Boolean) | Inspeksi Matriks Piksel</div>', unsafe_allow_html=True)

# Pastikan sampel gambar tersedia
if not os.path.exists("samples/sample_a.jpg") or not os.path.exists("samples/sample_b.jpg"):
    import generate_samples

# -------------------------------------------------------------
# SIDEBAR: 1. MENU INPUT & UPLOAD GAMBAR (FLEKSIBEL)
# -------------------------------------------------------------
st.sidebar.header("📂 1. Menu Input & Upload Citra")

input_mode = st.sidebar.radio(
    "Pilih Sumber Citra:",
    ["📁 Upload Gambar Sendiri (Bisa Diganti)", "🎨 Gunakan Gambar Sampel Bawaan"],
    index=0
)

img_a = None
img_b = None

if input_mode == "📁 Upload Gambar Sendiri (Bisa Diganti)":
    st.sidebar.markdown("**Unggah Citra A (Utama):**")
    file_a = st.sidebar.file_uploader("Pilih file Citra A", type=["png", "jpg", "jpeg", "bmp", "webp"], key="uploader_a")
    
    st.sidebar.markdown("**Unggah Citra B (Pasangan Operasi):**")
    file_b = st.sidebar.file_uploader("Pilih file Citra B", type=["png", "jpg", "jpeg", "bmp", "webp"], key="uploader_b")

    # Load Citra A
    if file_a is not None:
        pil_a = Image.open(file_a).convert("RGB")
        img_a = np.array(pil_a)
        st.sidebar.success(f"✅ Citra A: `{file_a.name}` ({img_a.shape[1]}x{img_a.shape[0]})")
    else:
        # Default fallback ke sample A jika belum upload
        img_a_bgr = cv2.imread("samples/sample_a.jpg")
        img_a = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2RGB)
        st.sidebar.info("ℹ️ Citra A: Menggunakan Sampel A (Upload file Anda di atas untuk mengganti).")

    # Load Citra B
    if file_b is not None:
        pil_b = Image.open(file_b).convert("RGB")
        img_b = np.array(pil_b)
        st.sidebar.success(f"✅ Citra B: `{file_b.name}` ({img_b.shape[1]}x{img_b.shape[0]})")
    else:
        # Default fallback ke sample B jika belum upload
        img_b_bgr = cv2.imread("samples/sample_b.jpg")
        img_b = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2RGB)
        st.sidebar.info("ℹ️ Citra B: Menggunakan Sampel B (Upload file Anda di atas untuk mengganti).")

else:
    # Gunakan preset sampel bawaan
    sample_choice_a = st.sidebar.selectbox("Pilih Sampel Citra A:", ["Sample A (Shapes & Gradient)", "Sample B (Texture)", "Masker Lingkaran"])
    if sample_choice_a == "Sample A (Shapes & Gradient)":
        img_a = cv2.cvtColor(cv2.imread("samples/sample_a.jpg"), cv2.COLOR_BGR2RGB)
    elif sample_choice_a == "Sample B (Texture)":
        img_a = cv2.cvtColor(cv2.imread("samples/sample_b.jpg"), cv2.COLOR_BGR2RGB)
    else:
        mask_g = cv2.imread("samples/sample_mask.png", cv2.IMREAD_GRAYSCALE)
        img_a = cv2.cvtColor(mask_g, cv2.COLOR_GRAY2RGB)

    sample_choice_b = st.sidebar.selectbox("Pilih Sampel Citra B:", ["Sample B (Texture)", "Sample A (Shapes & Gradient)", "Masker Lingkaran"])
    if sample_choice_b == "Sample B (Texture)":
        img_b = cv2.cvtColor(cv2.imread("samples/sample_b.jpg"), cv2.COLOR_BGR2RGB)
    elif sample_choice_b == "Sample A (Shapes & Gradient)":
        img_b = cv2.cvtColor(cv2.imread("samples/sample_a.jpg"), cv2.COLOR_BGR2RGB)
    else:
        mask_g = cv2.imread("samples/sample_mask.png", cv2.IMREAD_GRAYSCALE)
        img_b = cv2.cvtColor(mask_g, cv2.COLOR_GRAY2RGB)


# -------------------------------------------------------------
# SIDEBAR: 2. KATEGORI OPERASI
# -------------------------------------------------------------
st.sidebar.header("⚙️ 2. Kategori Operasi")
category = st.sidebar.radio(
    "Pilih Kategori:",
    ["🎯 Operasi Piksel (Aras Titik)", "➕ Operasi Aritmatika Citra", "⚡ Operasi Logika (Boolean)"]
)

img_result = img_a.copy()
calc_explanation = ""

# -------------------------------------------------------------
# KATEGORI 1: OPERASI PIKSEL (ARAS TITIK)
# -------------------------------------------------------------
if category == "🎯 Operasi Piksel (Aras Titik)":
    st.sidebar.subheader("Pengaturan Operasi Aras Titik")
    
    op_type = st.sidebar.selectbox(
        "Pilih Jenis Operasi Piksel:",
        [
            "Pencerahan (Brightness)",
            "Kontras (Contrast)",
            "Inversi / Negatif Citra",
            "Pengambangan (Thresholding)",
            "Koreksi Gamma (Power-Law)",
            "Transformasi Logaritmik",
            "Konversi Grayscale"
        ]
    )

    if op_type == "Pencerahan (Brightness)":
        b_val = st.sidebar.slider("Nilai Pencerahan (b):", -255, 255, 50, step=1)
        img_result = ops.adjust_brightness(img_a, b_val)
        calc_explanation = f"""**Rumus Pencerahan (Brightness):**
$$O(x,y) = \\min(\\max(I(x,y) + ({b_val}), 0), 255)$$
Setiap nilai intensitas piksel citra awal $I(x,y)$ ditambahkan konstanta $b = {b_val}$ lalu dipotong (clamp) ke rentang $[0, 255]$."""

    elif op_type == "Kontras (Contrast)":
        c_val = st.sidebar.slider("Faktor Kontras (c):", 0.0, 3.0, 1.5, step=0.05)
        img_result = ops.adjust_contrast(img_a, c_val)
        calc_explanation = f"""**Rumus Kontras (Peregangan terhadap titik tengah 128):**
$$O(x,y) = \\min(\\max(128 + {c_val:.2f} \\cdot (I(x,y) - 128), 0), 255)$$
Piksel di atas 128 menjadi lebih terang, dan piksel di bawah 128 menjadi lebih gelap."""

    elif op_type == "Inversi / Negatif Citra":
        img_result = ops.invert_image(img_a)
        calc_explanation = """**Rumus Negatif Citra (Invert):**
$$O(x,y) = 255 - I(x,y)$$
Membalikkan tingkat keabuan: area gelap menjadi terang dan area terang menjadi gelap."""

    elif op_type == "Pengambangan (Thresholding)":
        t_val = st.sidebar.slider("Ambang Batas (Threshold T):", 0, 255, 128, step=1)
        img_result = ops.threshold_image(img_a, t_val)
        calc_explanation = f"""**Rumus Thresholding Biner:**
$$O(x,y) = 255 \\quad \\text{{jika}} \\; I_{{gray}}(x,y) \\ge {t_val}, \\quad \\text{{dan}} \\; 0 \\; \\text{{jika}} \\; I_{{gray}}(x,y) < {t_val}$$
Menghasilkan citra biner (hitam dan putih) dengan pemisah ambang batas $T = {t_val}$."""

    elif op_type == "Koreksi Gamma (Power-Law)":
        gamma = st.sidebar.slider("Nilai Gamma (γ):", 0.1, 4.0, 0.5, step=0.05)
        img_result = ops.gamma_correction(img_a, gamma)
        calc_explanation = f"""**Rumus Koreksi Gamma:**
$$O(x,y) = 255 \\cdot \\left(\\frac{{I(x,y)}}{{255}}\\right)^{{1 / {gamma:.2f}}}$$
Nilai $\\gamma < 1$ mencerahkan daerah bayangan, sedangkan $\\gamma > 1$ menggelapkan citra."""

    elif op_type == "Transformasi Logaritmik":
        img_result = ops.log_transform(img_a)
        calc_explanation = """**Rumus Transformasi Logaritmik:**
$$O(x,y) = c \\cdot \\ln(1 + I(x,y)), \\quad c = \\frac{255}{\\ln(1 + \\max(I))}$$
Memperjelas detail pada area berintensitas rendah (gelap) dengan skala dinamis."""

    elif op_type == "Konversi Grayscale":
        gray_method = st.sidebar.selectbox("Metode Grayscale:", ["luminance", "average", "red_only", "green_only", "blue_only"])
        img_result = ops.to_grayscale(img_a, gray_method)
        if gray_method == "luminance":
            calc_explanation = """**Rumus Grayscale Luminance (Standar ITU-R BT.601):**
$$Y = 0.299 \\cdot R + 0.587 \\cdot G + 0.114 \\cdot B$$
Mempertimbangkan sensitivitas mata manusia tertinggi pada warna Hijau (Green)."""
        else:
            calc_explanation = """**Rumus Grayscale Rata-rata:**
$$Y = \\frac{R + G + B}{3}$$"""

# -------------------------------------------------------------
# KATEGORI 2: OPERASI ARITMATIKA CITRA
# -------------------------------------------------------------
elif category == "➕ Operasi Aritmatika Citra":
    st.sidebar.subheader("Pengaturan Operasi Aritmatika")
    
    arith_target = st.sidebar.radio("Operan Kedua:", ["Dengan Citra B", "Dengan Skalar (Konstanta)"])
    arith_op = st.sidebar.selectbox("Pilih Operasi Aritmatika:", [
        "Penjumlahan (Addition)",
        "Pengurangan (Subtraction)",
        "Perkalian (Multiplication)",
        "Pembagian (Division)",
        "Blending (Pencampuran Berbobot)"
    ])

    if arith_target == "Dengan Skalar (Konstanta)":
        scalar_val = st.sidebar.slider("Nilai Skalar (c):", 0, 255, 50, step=1)
        target_operand = scalar_val
        op_b_label = f"c = {scalar_val}"
    else:
        target_operand = img_b
        op_b_label = "Citra B"

    if arith_op == "Penjumlahan (Addition)":
        add_mode = st.sidebar.radio("Mode Penjumlahan:", ["Saturasi / Clamping", "Modulo 256"])
        mode_str = "clamp" if "Saturasi" in add_mode else "modulo"
        img_result = ops.image_add(img_a, target_operand, mode=mode_str)
        if mode_str == "clamp":
            calc_explanation = f"""**Rumus Penjumlahan (Saturasi):**
$$O(x,y) = \\min(A(x,y) + {op_b_label}, 255)$$
Jika hasil $> 255$, dipotong menjadi $255$ (putih maksimal)."""
        else:
            calc_explanation = f"""**Rumus Penjumlahan (Modulo):**
$$O(x,y) = (A(x,y) + {op_b_label}) \\pmod{{256}}$$
Hasil yang melebihi $255$ akan berputar kembali dari $0$ (*wrap-around*)."""

    elif arith_op == "Pengurangan (Subtraction)":
        use_abs = st.sidebar.checkbox("Gunakan Selisih Mutlak |A - B|", value=False)
        img_result = ops.image_subtract(img_a, target_operand, absolute=use_abs)
        if use_abs:
            calc_explanation = f"""**Rumus Selisih Mutlak (Difference):**
$$O(x,y) = |A(x,y) - {op_b_label}|$$
Berguna untuk mendeteksi perbedaan atau pergerakan objek."""
        else:
            calc_explanation = f"""**Rumus Pengurangan Standar (Clamping):**
$$O(x,y) = \\max(A(x,y) - {op_b_label}, 0)$$
Jika hasil $< 0$, dipotong menjadi $0$ (hitam pekat)."""

    elif arith_op == "Perkalian (Multiplication)":
        img_result = ops.image_multiply(img_a, target_operand)
        calc_explanation = f"""**Rumus Perkalian Citra:**
$$O(x,y) = \\frac{{A(x,y) \\cdot {op_b_label}}}{{255}}$$
Menghasilkan modulasi intensitas atau masking kecerahan."""

    elif arith_op == "Pembagian (Division)":
        img_result = ops.image_divide(img_a, target_operand)
        calc_explanation = f"""**Rumus Pembagian Citra:**
$$O(x,y) = \\frac{{A(x,y)}}{{{op_b_label} + 1}} \\cdot 255$$
Berguna untuk koreksi pencahayaan yang tidak merata (*shading correction*)."""

    elif arith_op == "Blending (Pencampuran Berbobot)":
        alpha = st.sidebar.slider("Bobot Alpha (α):", 0.0, 1.0, 0.5, step=0.05)
        img_result = ops.image_blend(img_a, img_b, alpha=alpha)
        calc_explanation = f"""**Rumus Citra Blending Linear:**
$$O(x,y) = {alpha:.2f} \\cdot A(x,y) + {1.0 - alpha:.2f} \\cdot B(x,y)$$
Menggabungkan dua citra secara transparan dengan bobot $\\alpha = {alpha:.2f}$."""

# -------------------------------------------------------------
# KATEGORI 3: OPERASI LOGIKA (BOOLEAN / BITWISE)
# -------------------------------------------------------------
elif category == "⚡ Operasi Logika (Boolean)":
    st.sidebar.subheader("Pengaturan Operasi Logika Boolean")
    
    logic_source = st.sidebar.radio("Sumber Masker / Citra B:", ["Masker Geometris Biner", "Citra B"])
    
    if logic_source == "Masker Geometris Biner":
        mask_type = st.sidebar.selectbox("Bentuk Masker:", ["circle", "rect", "diamond", "checker"])
        mask_size = st.sidebar.slider("Ukuran Masker:", 20, 200, 100, step=5)
        second_img = ops.create_shape_mask(img_a.shape, mask_type=mask_type, param=mask_size)
    else:
        second_img = ops.resize_match(img_b, img_a.shape)

    logic_op = st.sidebar.selectbox("Pilih Operasi Logika:", ["AND", "OR", "NOT", "XOR", "NAND", "NOR"])

    if logic_op == "AND":
        img_result = ops.boolean_and(img_a, second_img)
        calc_explanation = """**Operasi Bitwise AND (∧):**
$$O(x,y) = A(x,y) \\land B(x,y)$$
Hanya bagian citra yang bernilai $1$ (putih) pada masker yang akan dipertahankan (*ROI Masking*)."""
    elif logic_op == "OR":
        img_result = ops.boolean_or(img_a, second_img)
        calc_explanation = """**Operasi Bitwise OR (∨):**
$$O(x,y) = A(x,y) \\lor B(x,y)$$
Menggabungkan area piksel dari kedua citra."""
    elif logic_op == "NOT":
        img_result = ops.boolean_not(img_a)
        calc_explanation = """**Operasi Bitwise NOT (¬):**
$$O(x,y) = \\sim A(x,y) = 255 - A(x,y)$$
Membalikkan seluruh bit representasi citra."""
    elif logic_op == "XOR":
        img_result = ops.boolean_xor(img_a, second_img)
        calc_explanation = """**Operasi Bitwise XOR (⊕):**
$$O(x,y) = A(x,y) \\oplus B(x,y)$$
Bernilai $0$ jika kedua citra sama, dan bernilai $>0$ jika terdapat perbedaan."""
    elif logic_op == "NAND":
        img_result = ops.boolean_nand(img_a, second_img)
        calc_explanation = """**Operasi Bitwise NAND:**
$$O(x,y) = \\sim (A(x,y) \\land B(x,y))$$"""
    elif logic_op == "NOR":
        img_result = ops.boolean_nor(img_a, second_img)
        calc_explanation = """**Operasi Bitwise NOR:**
$$O(x,y) = \\sim (A(x,y) \\lor B(x,y))$$"""


# -------------------------------------------------------------
# MAIN DISPLAY: 3 TAMPILAN CITRA
# -------------------------------------------------------------
col_img1, col_img2, col_img3 = st.columns(3)

with col_img1:
    st.subheader("1️⃣ Citra Asli (A)")
    st.image(img_a, use_column_width=True, caption=f"Citra A ({img_a.shape[1]}x{img_a.shape[0]})")

with col_img2:
    st.subheader("2️⃣ Citra B / Masker")
    if category == "⚡ Operasi Logika (Boolean)" and logic_source == "Masker Geometris Biner":
        st.image(second_img, use_column_width=True, caption=f"Masker Biner ({mask_type})")
    else:
        st.image(img_b, use_column_width=True, caption=f"Citra B ({img_b.shape[1]}x{img_b.shape[0]})")

with col_img3:
    st.subheader("3️⃣ Citra Hasil Pemrosesan")
    st.image(img_result, use_column_width=True, caption="Citra Hasil Operasi")


# -------------------------------------------------------------
# PANEL INSPEKSI NILAI MATRIKS PIKSEL INTERAKTIF
# -------------------------------------------------------------
st.markdown("---")
st.header("🔍 Panel Inspeksi Nilai Matriks Piksel")
st.write("Pilih koordinat piksel $(X, Y)$ di bawah ini untuk melihat rincian warna dan matriks tetangganya:")

h, w = img_a.shape[:2]
c_pos1, c_pos2, c_pos3, c_pos4 = st.columns(4)

with c_pos1:
    x_sel = st.slider("Koordinat X (Kolom):", 0, w - 1, w // 2)
with c_pos2:
    y_sel = st.slider("Koordinat Y (Baris):", 0, h - 1, h // 2)
with c_pos3:
    mat_size = st.selectbox("Ukuran Matriks:", [3, 5], index=0)
with c_pos4:
    mat_source = st.selectbox("Ambil dari Citra:", ["Citra Asli (A)", "Citra Hasil Pemrosesan", "Citra B / Masker"])

# Tentukan sumber citra inspeksi
if mat_source == "Citra Asli (A)":
    inspect_img = img_a
elif mat_source == "Citra B / Masker":
    inspect_img = second_img if (category == "⚡ Operasi Logika (Boolean)" and logic_source == "Masker Geometris Biner") else ops.resize_match(img_b, img_a.shape)
else:
    inspect_img = img_result

info = ops.get_pixel_neighborhood(inspect_img, x_sel, y_sel, size=mat_size)

col_detail, col_mat, col_math = st.columns([1.2, 1.5, 1.3])

with col_detail:
    st.subheader("📍 Nilai Piksel Pusat")
    st.markdown(f"""
    <div class="metric-card">
        <b>Sumber:</b> {mat_source}<br>
        <b>Titik (X, Y):</b> ({x_sel}, {y_sel})<br>
        <b>RGB:</b> <span style="color:rgb({info['r']},{info['g']},{info['b']}); font-weight:bold;">[{info['r']}, {info['g']}, {info['b']}]</span><br>
        <b>HEX:</b> <code>{info['hex']}</code><br>
        <b>Grayscale:</b> {info['gray']}<br>
        <hr style="margin:6px 0;">
        <b>Biner 8-bit:</b><br>
        <code>R: {info['binary_r']}</code><br>
        <code>G: {info['binary_g']}</code><br>
        <code>B: {info['binary_b']}</code>
    </div>
    """, unsafe_allow_html=True)

with col_mat:
    st.subheader(f"🔢 Matriks Tetangga {mat_size}x{mat_size}")
    sub_chan = st.radio("Pilih Channel:", ["Grayscale", "Red (R)", "Green (G)", "Blue (B)"], horizontal=True)
    
    if sub_chan == "Red (R)":
        mat_display = np.array(info["mat_r"])
        cmap_name = "Reds"
    elif sub_chan == "Green (G)":
        mat_display = np.array(info["mat_g"])
        cmap_name = "Greens"
    elif sub_chan == "Blue (B)":
        mat_display = np.array(info["mat_b"])
        cmap_name = "Blues"
    else:
        mat_display = np.array(info["mat_gray"])
        cmap_name = "gray"

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(mat_display, cmap=cmap_name, vmin=0, vmax=255)
    
    cx = info["center_offset"]["cx"]
    cy = info["center_offset"]["cy"]

    for i in range(mat_display.shape[0]):
        for j in range(mat_display.shape[1]):
            val = mat_display[i, j]
            if i == cy and j == cx:
                text_color = "yellow" if val < 128 else "red"
                ax.text(j, i, f"★{val}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=11)
            else:
                text_color = "white" if val < 128 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=text_color, fontsize=9)

    ax.set_xticks(range(mat_display.shape[1]))
    ax.set_yticks(range(mat_display.shape[0]))
    ax.set_xticklabels([f"x={info['bounds']['x_min']+k}" for k in range(mat_display.shape[1])], fontsize=7)
    ax.set_yticklabels([f"y={info['bounds']['y_min']+k}" for k in range(mat_display.shape[0])], fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)

with col_math:
    st.subheader("📐 Penjelasan & Rumus")
    st.markdown(calc_explanation)

    p_a_val = img_a[y_sel, x_sel]
    p_res_val = img_result[y_sel, x_sel]
    st.markdown(f"""
    **Perubahan Nilai pada Titik ({x_sel}, {y_sel}):**
    - **Citra Awal (A):** RGB `{p_a_val.tolist()}`
    - **Citra Hasil:** RGB `{p_res_val.tolist()}`
    """)
