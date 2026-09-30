#!/usr/bin/env python3
"""
app_gui.py
Aplikasi GUI Desktop Pengolahan Citra Digital - Praktikum Pertemuan 2
Materi:
1. Operasi Piksel (Aras Titik)
2. Operasi Aritmatika Citra
3. Operasi Logika (Boolean) Citra
+ Fitur Inspeksi Nilai Matriks Piksel Interaktif (Klik pada gambar)
"""

import os
import sys

# Silence macOS Tkinter deprecation warning
os.environ["TK_SILENCE_DEPRECATION"] = "1"

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Import fungsi operasi citra
import image_operations as ops


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pengolahan Citra Digital - Pertemuan 2 | Praktikum Aras Titik, Aritmatika & Logika")
        self.root.geometry("1300x860")
        self.root.minsize(1100, 750)

        # Style configuration
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Data Citra
        self.img_a_original = None      # RGB numpy array (Citra Utama)
        self.img_b_original = None      # RGB numpy array (Citra Kedua)
        self.img_result = None          # RGB numpy array (Hasil Operasi)
        self.display_scale = 1.0        # Skala tampilan canvas ke koordinat asli
        self.selected_coord = None      # (x, y) koordinat piksel yang dipilih
        self.last_clicked_source = "A"  # 'A', 'B', atau 'Result'

        # Variabel Kontrol Operasi Piksel
        self.var_brightness = tk.IntVar(value=0)
        self.var_contrast = tk.DoubleVar(value=1.0)
        self.var_gamma = tk.DoubleVar(value=1.0)
        self.var_threshold = tk.IntVar(value=128)
        self.var_invert = tk.BooleanVar(value=False)
        self.var_log_transform = tk.BooleanVar(value=False)
        self.var_grayscale_method = tk.StringVar(value="none")

        # Variabel Kontrol Operasi Aritmatika
        self.var_arith_op = tk.StringVar(value="add")
        self.var_arith_target = tk.StringVar(value="image_b")  # 'image_b' atau 'scalar'
        self.var_arith_scalar = tk.DoubleVar(value=50.0)
        self.var_blend_alpha = tk.DoubleVar(value=0.5)
        self.var_add_mode = tk.StringVar(value="clamp")  # 'clamp' atau 'modulo'
        self.var_sub_abs = tk.BooleanVar(value=False)

        # Variabel Kontrol Operasi Logika
        self.var_logic_op = tk.StringVar(value="and")
        self.var_logic_source = tk.StringVar(value="shape_mask")  # 'image_b' atau 'shape_mask'
        self.var_mask_type = tk.StringVar(value="circle")
        self.var_mask_size = tk.IntVar(value=100)

        # Variabel Inspeksi Matriks
        self.var_matrix_size = tk.IntVar(value=3)  # 3x3 atau 5x5
        self.var_matrix_channel = tk.StringVar(value="Grayscale")

        self._build_ui()
        self._load_default_samples()

    # -------------------------------------------------------------
    # UI CONSTRUCTION
    # -------------------------------------------------------------
    def _build_ui(self):
        # Top Header & Action Bar
        top_bar = tk.Frame(self.root, bg="#1E293B", padx=12, pady=10)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(
            top_bar,
            text="🖼️ Pengolahan Citra Digital - Praktikum Pertemuan 2",
            font=("Helvetica", 16, "bold"),
            fg="#F8FAFC",
            bg="#1E293B"
        )
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = tk.Label(
            top_bar,
            text="(Aras Titik | Aritmatika Citra | Logika Boolean | Inspeksi Matriks Piksel)",
            font=("Helvetica", 11, "italic"),
            fg="#94A3B8",
            bg="#1E293B"
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=10)

        # Buttons on Top Bar
        btn_samples = tk.Button(
            top_bar, text="📁 Muat Contoh Bawaan", bg="#3B82F6", fg="white",
            font=("Helvetica", 10, "bold"), relief=tk.FLAT, padx=8, pady=4,
            command=self._load_default_samples, cursor="hand2"
        )
        btn_samples.pack(side=tk.RIGHT, padx=5)

        btn_save = tk.Button(
            top_bar, text="💾 Simpan Hasil Citra", bg="#10B981", fg="white",
            font=("Helvetica", 10, "bold"), relief=tk.FLAT, padx=8, pady=4,
            command=self._save_result_image, cursor="hand2"
        )
        btn_save.pack(side=tk.RIGHT, padx=5)

        # Main Workspace: Left Controls & Right Views
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # LEFT PANEL: Control Notebook (Tabs)
        left_frame = ttk.Frame(main_paned, width=380)
        main_paned.add(left_frame, weight=0)

        self._build_control_tabs(left_frame)

        # RIGHT PANEL: Images Display + Pixel Inspector
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        self._build_display_and_inspector(right_frame)

        # Bottom Status Bar
        self.status_bar = tk.Label(
            self.root, text="Siap. Silakan pilih operasi atau klik pada gambar untuk inspeksi piksel.",
            bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Helvetica", 9), bg="#F1F5F9", fg="#334155", padx=6, pady=3
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_control_tabs(self, parent):
        # File Loader Section
        loader_group = ttk.LabelFrame(parent, text="📂 Input Citra Digital", padding=8)
        loader_group.pack(fill=tk.X, padx=5, pady=5)

        # Row 1: Citra A
        f_load_a = ttk.Frame(loader_group)
        f_load_a.pack(fill=tk.X, pady=2)
        ttk.Label(f_load_a, text="Citra A (Utama):", font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        btn_load_a = ttk.Button(f_load_a, text="Pilih File...", command=self._load_image_a)
        btn_load_a.pack(side=tk.RIGHT)

        # Row 2: Citra B
        f_load_b = ttk.Frame(loader_group)
        f_load_b.pack(fill=tk.X, pady=2)
        ttk.Label(f_load_b, text="Citra B (Arit/Log):", font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        btn_load_b = ttk.Button(f_load_b, text="Pilih File...", command=self._load_image_b)
        btn_load_b.pack(side=tk.RIGHT)

        # Notebook for Operations
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # TAB 1: Operasi Piksel (Aras Titik)
        tab_pixel = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab_pixel, text="🎯 1. Aras Titik (Piksel)")
        self._build_tab_pixel(tab_pixel)

        # TAB 2: Operasi Aritmatika
        tab_arith = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab_arith, text="➕ 2. Aritmatika")
        self._build_tab_arith(tab_arith)

        # TAB 3: Operasi Logika (Boolean)
        tab_logic = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab_logic, text="⚡ 3. Logika (Boolean)")
        self._build_tab_logic(tab_logic)

        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self.apply_current_operation())

    # ---------------- TAB 1: OPERASI PIKSEL ----------------
    def _build_tab_pixel(self, tab):
        scroll_canvas = tk.Canvas(tab, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=scroll_canvas.yview)
        inner = ttk.Frame(scroll_canvas)

        inner.bind("<Configure>", lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
        scroll_canvas.create_window((0, 0), window=inner, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 1. Pencerahan (Brightness)
        grp_bright = ttk.LabelFrame(inner, text="☀️ Pencerahan (Brightness: f(x,y) + b)", padding=6)
        grp_bright.pack(fill=tk.X, pady=4)
        f_b = ttk.Frame(grp_bright)
        f_b.pack(fill=tk.X)
        self.lbl_bright_val = ttk.Label(f_b, text="0", width=4)
        self.lbl_bright_val.pack(side=tk.RIGHT)
        scale_bright = ttk.Scale(
            f_b, from_=-255, to=255, variable=self.var_brightness, orient=tk.HORIZONTAL,
            command=lambda v: (self.lbl_bright_val.config(text=str(int(float(v)))), self.apply_current_operation())
        )
        scale_bright.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 2. Kontras (Contrast)
        grp_contrast = ttk.LabelFrame(inner, text="🌓 Kontras (Contrast: c · f(x,y))", padding=6)
        grp_contrast.pack(fill=tk.X, pady=4)
        f_c = ttk.Frame(grp_contrast)
        f_c.pack(fill=tk.X)
        self.lbl_contrast_val = ttk.Label(f_c, text="1.0x", width=5)
        self.lbl_contrast_val.pack(side=tk.RIGHT)
        scale_contrast = ttk.Scale(
            f_c, from_=0.0, to=3.0, variable=self.var_contrast, orient=tk.HORIZONTAL,
            command=lambda v: (self.lbl_contrast_val.config(text=f"{float(v):.2f}x"), self.apply_current_operation())
        )
        scale_contrast.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 3. Negatif / Invert Citra
        grp_inv = ttk.LabelFrame(inner, text="🔄 Inversi (Citra Negatif: 255 - f(x,y))", padding=6)
        grp_inv.pack(fill=tk.X, pady=4)
        chk_inv = ttk.Checkbutton(
            grp_inv, text="Aktifkan Negatif Citra", variable=self.var_invert,
            command=self.apply_current_operation
        )
        chk_inv.pack(anchor=tk.W)

        # 4. Pengambangan (Thresholding / Binarisasi)
        grp_thresh = ttk.LabelFrame(inner, text="🔲 Pengambangan (Thresholding: T)", padding=6)
        grp_thresh.pack(fill=tk.X, pady=4)
        f_t = ttk.Frame(grp_thresh)
        f_t.pack(fill=tk.X)
        self.lbl_thresh_val = ttk.Label(f_t, text="128", width=4)
        self.lbl_thresh_val.pack(side=tk.RIGHT)
        scale_thresh = ttk.Scale(
            f_t, from_=0, to=255, variable=self.var_threshold, orient=tk.HORIZONTAL,
            command=lambda v: (self.lbl_thresh_val.config(text=str(int(float(v)))), self.apply_current_operation())
        )
        scale_thresh.pack(side=tk.LEFT, fill=tk.X, expand=True)
        btn_thresh = ttk.Button(grp_thresh, text="Terapkan Biner Threshold", command=self._set_threshold_mode)
        btn_thresh.pack(fill=tk.X, pady=2)

        # 5. Koreksi Gamma (Power Law)
        grp_gamma = ttk.LabelFrame(inner, text="🌈 Koreksi Gamma (c · f(x,y)^γ)", padding=6)
        grp_gamma.pack(fill=tk.X, pady=4)
        f_g = ttk.Frame(grp_gamma)
        f_g.pack(fill=tk.X)
        self.lbl_gamma_val = ttk.Label(f_g, text="1.0", width=4)
        self.lbl_gamma_val.pack(side=tk.RIGHT)
        scale_gamma = ttk.Scale(
            f_g, from_=0.1, to=4.0, variable=self.var_gamma, orient=tk.HORIZONTAL,
            command=lambda v: (self.lbl_gamma_val.config(text=f"{float(v):.2f}"), self.apply_current_operation())
        )
        scale_gamma.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 6. Transformasi Log & Grayscale
        grp_extra = ttk.LabelFrame(inner, text="⚙️ Konversi & Log Transform", padding=6)
        grp_extra.pack(fill=tk.X, pady=4)
        chk_log = ttk.Checkbutton(
            grp_extra, text="Transformasi Logaritmik c·log(1+r)",
            variable=self.var_log_transform, command=self.apply_current_operation
        )
        chk_log.pack(anchor=tk.W, pady=2)

        f_gray = ttk.Frame(grp_extra)
        f_gray.pack(fill=tk.X, pady=2)
        ttk.Label(f_gray, text="Grayscale:").pack(side=tk.LEFT)
        cb_gray = ttk.Combobox(
            f_gray, textvariable=self.var_grayscale_method,
            values=["none", "luminance", "average", "red_only", "green_only", "blue_only"],
            state="readonly", width=12
        )
        cb_gray.pack(side=tk.RIGHT)
        cb_gray.bind("<<ComboboxSelected>>", lambda e: self.apply_current_operation())

        # Reset Button
        btn_reset = ttk.Button(inner, text="↺ Reset Operasi Piksel", command=self._reset_pixel_params)
        btn_reset.pack(fill=tk.X, pady=8)

    def _set_threshold_mode(self):
        self.var_grayscale_method.set("luminance")
        self.apply_current_operation()

    def _reset_pixel_params(self):
        self.var_brightness.set(0)
        self.lbl_bright_val.config(text="0")
        self.var_contrast.set(1.0)
        self.lbl_contrast_val.config(text="1.0x")
        self.var_gamma.set(1.0)
        self.lbl_gamma_val.config(text="1.0")
        self.var_threshold.set(128)
        self.lbl_thresh_val.config(text="128")
        self.var_invert.set(False)
        self.var_log_transform.set(False)
        self.var_grayscale_method.set("none")
        self.apply_current_operation()

    # ---------------- TAB 2: OPERASI ARITMATIKA ----------------
    def _build_tab_arith(self, tab):
        grp_target = ttk.LabelFrame(tab, text="🎯 Sumber Pasangan Operasi", padding=6)
        grp_target.pack(fill=tk.X, pady=4)

        rb_img = ttk.Radiobutton(
            grp_target, text="Dengan Citra B", variable=self.var_arith_target,
            value="image_b", command=self.apply_current_operation
        )
        rb_img.pack(anchor=tk.W)

        rb_scal = ttk.Radiobutton(
            grp_target, text="Dengan Nilai Skalar (Konstanta c)", variable=self.var_arith_target,
            value="scalar", command=self.apply_current_operation
        )
        rb_scal.pack(anchor=tk.W)

        f_sc = ttk.Frame(grp_target)
        f_sc.pack(fill=tk.X, pady=4)
        ttk.Label(f_sc, text="Nilai Skalar (c):").pack(side=tk.LEFT)
        scale_sc = ttk.Scale(
            f_sc, from_=0, to=255, variable=self.var_arith_scalar, orient=tk.HORIZONTAL,
            command=lambda v: (self.lbl_sc_val.config(text=str(int(float(v)))), self.apply_current_operation())
        )
        scale_sc.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.lbl_sc_val = ttk.Label(f_sc, text="50", width=4)
        self.lbl_sc_val.pack(side=tk.RIGHT)

        # Operations
        grp_ops = ttk.LabelFrame(tab, text="🔢 Pilihan Operasi Aritmatika", padding=6)
        grp_ops.pack(fill=tk.X, pady=4)

        arith_list = [
            ("add", "➕ Penjumlahan (A + B atau A + c)"),
            ("sub", "➖ Pengurangan (A - B atau A - c)"),
            ("mul", "✖️ Perkalian (A · B / 255 atau A · c)"),
            ("div", "➗ Pembagian ((A / (B+1)) · 255 atau A / c)"),
            ("blend", "🎨 Blending Bobot (α·A + (1-α)·B)"),
        ]
        for val, txt in arith_list:
            ttk.Radiobutton(
                grp_ops, text=txt, variable=self.var_arith_op,
                value=val, command=self.apply_current_operation
            ).pack(anchor=tk.W, pady=2)

        # Blending Alpha Slider
        grp_blend = ttk.LabelFrame(tab, text="🎚️ Parameter Blending (Alpha α)", padding=6)
        grp_blend.pack(fill=tk.X, pady=4)
        f_al = ttk.Frame(grp_blend)
        f_al.pack(fill=tk.X)
        self.lbl_alpha_val = ttk.Label(f_al, text="0.50", width=5)
        self.lbl_alpha_val.pack(side=tk.RIGHT)
        scale_alpha = ttk.Scale(
            f_al, from_=0.0, to=1.0, variable=self.var_blend_alpha, orient=tk.HORIZONTAL,
            command=lambda v: (self.lbl_alpha_val.config(text=f"{float(v):.2f}"), self.apply_current_operation())
        )
        scale_alpha.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Options for Addition & Subtraction
        grp_opt = ttk.LabelFrame(tab, text="⚙️ Opsi Penjumlahan / Pengurangan", padding=6)
        grp_opt.pack(fill=tk.X, pady=4)
        ttk.Radiobutton(
            grp_opt, text="Saturasi / Clamping [0..255]", variable=self.var_add_mode,
            value="clamp", command=self.apply_current_operation
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            grp_opt, text="Modulo 256 (Wrap-around)", variable=self.var_add_mode,
            value="modulo", command=self.apply_current_operation
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            grp_opt, text="Selisih Mutlak |A - B| (Difference)", variable=self.var_sub_abs,
            command=self.apply_current_operation
        ).pack(anchor=tk.W, pady=2)

    # ---------------- TAB 3: OPERASI LOGIKA ----------------
    def _build_tab_logic(self, tab):
        grp_src = ttk.LabelFrame(tab, text="🎭 Sumber Citra Kedua / Masker", padding=6)
        grp_src.pack(fill=tk.X, pady=4)

        ttk.Radiobutton(
            grp_src, text="Gunakan Masker Geometri Biner", variable=self.var_logic_source,
            value="shape_mask", command=self.apply_current_operation
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            grp_src, text="Gunakan Citra B", variable=self.var_logic_source,
            value="image_b", command=self.apply_current_operation
        ).pack(anchor=tk.W)

        # Shape mask selection
        f_shp = ttk.Frame(grp_src)
        f_shp.pack(fill=tk.X, pady=3)
        ttk.Label(f_shp, text="Bentuk Masker:").pack(side=tk.LEFT)
        cb_mask = ttk.Combobox(
            f_shp, textvariable=self.var_mask_type,
            values=["circle", "rect", "diamond", "checker"],
            state="readonly", width=12
        )
        cb_mask.pack(side=tk.RIGHT)
        cb_mask.bind("<<ComboboxSelected>>", lambda e: self.apply_current_operation())

        f_msz = ttk.Frame(grp_src)
        f_msz.pack(fill=tk.X, pady=2)
        ttk.Label(f_msz, text="Ukuran Masker:").pack(side=tk.LEFT)
        scale_msz = ttk.Scale(
            f_msz, from_=20, to=200, variable=self.var_mask_size, orient=tk.HORIZONTAL,
            command=lambda v: (self.lbl_msz_val.config(text=str(int(float(v)))), self.apply_current_operation())
        )
        scale_msz.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.lbl_msz_val = ttk.Label(f_msz, text="100", width=4)
        self.lbl_msz_val.pack(side=tk.RIGHT)

        # Logic Operations
        grp_lops = ttk.LabelFrame(tab, text="⚡ Operasi Bitwise Boolean", padding=6)
        grp_lops.pack(fill=tk.X, pady=4)

        logic_list = [
            ("and", "AND  (A ∧ B) -> Masking & Irisan"),
            ("or",  "OR   (A ∨ B) -> Penggabungan"),
            ("not", "NOT  (¬A)    -> Pembalik Bit"),
            ("xor", "XOR  (A ⊕ B) -> Deteksi Perubahan"),
            ("nand","NAND (¬(A ∧ B))"),
            ("nor", "NOR  (¬(A ∨ B))"),
        ]
        for val, txt in logic_list:
            ttk.Radiobutton(
                grp_lops, text=txt, variable=self.var_logic_op,
                value=val, command=self.apply_current_operation
            ).pack(anchor=tk.W, pady=2)

    # ---------------- DISPLAY & INSPECTOR PANEL ----------------
    def _build_display_and_inspector(self, parent):
        vpaned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        vpaned.pack(fill=tk.BOTH, expand=True)

        # TOP: Canvas Frame with 3 views (Citra A, Citra B/Mask, Hasil)
        frame_views = ttk.LabelFrame(vpaned, text="👁️ Tampilan Citra (Klik titik piksel pada citra manapun untuk melihat nilai matriks)", padding=6)
        vpaned.add(frame_views, weight=3)

        views_container = ttk.Frame(frame_views)
        views_container.pack(fill=tk.BOTH, expand=True)

        # View 1: Citra Asli (A)
        self.frame_v1 = ttk.LabelFrame(views_container, text="1️⃣ Citra Asli (A)")
        self.frame_v1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        self.canvas_a = tk.Canvas(self.frame_v1, bg="#0F172A", cursor="crosshair")
        self.canvas_a.pack(fill=tk.BOTH, expand=True)
        self.canvas_a.bind("<Button-1>", lambda e: self._on_canvas_click(e, "A"))

        # View 2: Citra B / Masker
        self.frame_v2 = ttk.LabelFrame(views_container, text="2️⃣ Citra B / Masker")
        self.frame_v2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        self.canvas_b = tk.Canvas(self.frame_v2, bg="#0F172A", cursor="crosshair")
        self.canvas_b.pack(fill=tk.BOTH, expand=True)
        self.canvas_b.bind("<Button-1>", lambda e: self._on_canvas_click(e, "B"))

        # View 3: Citra Hasil
        self.frame_v3 = ttk.LabelFrame(views_container, text="3️⃣ Citra Hasil Pemrosesan")
        self.frame_v3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        self.canvas_res = tk.Canvas(self.frame_v3, bg="#0F172A", cursor="crosshair")
        self.canvas_res.pack(fill=tk.BOTH, expand=True)
        self.canvas_res.bind("<Button-1>", lambda e: self._on_canvas_click(e, "Result"))

        # BOTTOM: Pixel & Matrix Inspector Panel
        frame_inspector = ttk.LabelFrame(vpaned, text="🔍 Panel Inspeksi Nilai Matriks Piksel", padding=8)
        vpaned.add(frame_inspector, weight=2)

        self._build_inspector_content(frame_inspector)

    def _build_inspector_content(self, parent):
        col1 = ttk.Frame(parent, width=280)
        col1.pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(col1, text="📍 Detail Piksel Terpilih:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)

        # Standard tk.Label for styled colored text
        self.lbl_inspect_src = tk.Label(col1, text="Sumber: Citra Asli (A)", font=("Helvetica", 9, "italic"), fg="#0284C7")
        self.lbl_inspect_src.pack(anchor=tk.W, pady=1)

        self.lbl_inspect_coord = ttk.Label(col1, text="Koordinat (X, Y): (150, 150)", font=("Helvetica", 9, "bold"))
        self.lbl_inspect_coord.pack(anchor=tk.W, pady=2)

        # Color preview swatch + RGB label
        f_swatch = ttk.Frame(col1)
        f_swatch.pack(anchor=tk.W, pady=3)
        self.swatch_preview = tk.Canvas(f_swatch, width=24, height=24, bg="#FFFF00", highlightthickness=1, highlightbackground="#64748B")
        self.swatch_preview.pack(side=tk.LEFT, padx=4)
        self.lbl_inspect_rgb = ttk.Label(f_swatch, text="RGB: [255, 255, 0]\nHEX: #FFFF00", font=("Consolas", 9))
        self.lbl_inspect_rgb.pack(side=tk.LEFT)

        self.lbl_inspect_gray = ttk.Label(col1, text="Grayscale (Intensitas): 225", font=("Consolas", 9))
        self.lbl_inspect_gray.pack(anchor=tk.W, pady=1)

        self.lbl_inspect_bin = tk.Label(col1, text="Biner: R=11111111 G=11111111 B=00000000", font=("Consolas", 8), fg="#475569")
        self.lbl_inspect_bin.pack(anchor=tk.W, pady=1)

        # Col 2: Matrix Grid
        col2 = ttk.LabelFrame(parent, text="🔢 Matriks Tetangga (Sub-Matrix)", padding=6)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8)

        # Matrix Controls (Channel & Size)
        f_mat_ctrl = ttk.Frame(col2)
        f_mat_ctrl.pack(fill=tk.X, pady=2)

        ttk.Label(f_mat_ctrl, text="Ukuran:").pack(side=tk.LEFT)
        for sz in [3, 5]:
            ttk.Radiobutton(
                f_mat_ctrl, text=f"{sz}x{sz}", variable=self.var_matrix_size,
                value=sz, command=self._update_matrix_display
            ).pack(side=tk.LEFT, padx=2)

        ttk.Label(f_mat_ctrl, text="  Channel:").pack(side=tk.LEFT)
        cb_mchan = ttk.Combobox(
            f_mat_ctrl, textvariable=self.var_matrix_channel,
            values=["Grayscale", "Red (R)", "Green (G)", "Blue (B)"],
            state="readonly", width=12
        )
        cb_mchan.pack(side=tk.LEFT, padx=3)
        cb_mchan.bind("<<ComboboxSelected>>", lambda e: self._update_matrix_display())

        # Grid container for matrix cells
        self.grid_container = tk.Frame(col2, bg="#F8FAFC", bd=1, relief=tk.SOLID)
        self.grid_container.pack(fill=tk.BOTH, expand=True, pady=4)

        # Col 3: Mathematical Operation Calculation at Selected Pixel
        col3 = ttk.LabelFrame(parent, text="📐 Perhitungan Matematika Piksel Terpilih", padding=6)
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8)

        self.txt_math_calc = tk.Text(col3, height=8, width=32, font=("Consolas", 9), bg="#F8FAFC", wrap=tk.WORD, bd=0)
        self.txt_math_calc.pack(fill=tk.BOTH, expand=True)

    # -------------------------------------------------------------
    # IMAGE LOADING & SAVING
    # -------------------------------------------------------------
    def _load_default_samples(self):
        sample_a_path = "samples/sample_a.jpg"
        sample_b_path = "samples/sample_b.jpg"

        if not os.path.exists(sample_a_path) or not os.path.exists(sample_b_path):
            import generate_samples

        if os.path.exists(sample_a_path):
            img_bgr = cv2.imread(sample_a_path)
            self.img_a_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if os.path.exists(sample_b_path):
            img_bgr = cv2.imread(sample_b_path)
            self.img_b_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        self.selected_coord = (self.img_a_original.shape[1] // 2, self.img_a_original.shape[0] // 2)
        self.apply_current_operation()
        self.status_bar.config(text="Berhasil memuat citra contoh (Sample A & Sample B).")

    def _load_image_a(self):
        path = filedialog.askopenfilename(
            title="Pilih Citra A (Utama)",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff")]
        )
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.img_a_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.selected_coord = (self.img_a_original.shape[1] // 2, self.img_a_original.shape[0] // 2)
                self.apply_current_operation()
                self.status_bar.config(text=f"Citra A dimuat: {os.path.basename(path)} ({self.img_a_original.shape[1]}x{self.img_a_original.shape[0]})")
            else:
                messagebox.showerror("Error", "Gagal membaca file gambar!")

    def _load_image_b(self):
        path = filedialog.askopenfilename(
            title="Pilih Citra B (Pasangan Operasi)",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff")]
        )
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.img_b_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.apply_current_operation()
                self.status_bar.config(text=f"Citra B dimuat: {os.path.basename(path)} ({self.img_b_original.shape[1]}x{self.img_b_original.shape[0]})")
            else:
                messagebox.showerror("Error", "Gagal membaca file gambar!")

    def _save_result_image(self):
        if self.img_result is None:
            messagebox.showwarning("Peringatan", "Belum ada citra hasil pemrosesan.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("BMP Image", "*.bmp")]
        )
        if path:
            bgr = cv2.cvtColor(self.img_result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)
            messagebox.showinfo("Sukses", f"Citra hasil berhasil disimpan ke:\n{path}")

    # -------------------------------------------------------------
    # CORE PROCESSING ENGINE
    # -------------------------------------------------------------
    def apply_current_operation(self):
        if self.img_a_original is None:
            return

        active_tab_idx = self.notebook.index(self.notebook.select())
        img_a = self.img_a_original.copy()
        math_desc = ""

        # TAB 0: Operasi Piksel
        if active_tab_idx == 0:
            res = img_a
            # 1. Grayscale
            gray_m = self.var_grayscale_method.get()
            if gray_m != "none":
                res = ops.to_grayscale(res, gray_m)
            # 2. Brightness
            b_val = self.var_brightness.get()
            if b_val != 0:
                res = ops.adjust_brightness(res, b_val)
            # 3. Contrast
            c_val = self.var_contrast.get()
            if c_val != 1.0:
                res = ops.adjust_contrast(res, c_val)
            # 4. Invert
            if self.var_invert.get():
                res = ops.invert_image(res)
            # 5. Gamma
            g_val = self.var_gamma.get()
            if abs(g_val - 1.0) > 0.01:
                res = ops.gamma_correction(res, g_val)
            # 6. Log Transform
            if self.var_log_transform.get():
                res = ops.log_transform(res)
            # 7. Threshold
            t_val = self.var_threshold.get()
            if gray_m == "luminance" and t_val != 128:
                res = ops.threshold_image(res, t_val)

            self.img_result = res
            math_desc = f"OPERASI PIKSEL (ARAS TITIK):\n• Pencerahan (b): {b_val}\n• Kontras (c): {c_val:.2f}\n• Gamma (γ): {g_val:.2f}\n• Negatif: {self.var_invert.get()}\n• Grayscale: {gray_m}"

        # TAB 1: Operasi Aritmatika
        elif active_tab_idx == 1:
            op = self.var_arith_op.get()
            target_type = self.var_arith_target.get()

            if target_type == "image_b":
                b_source = self.img_b_original if self.img_b_original is not None else self.img_a_original
            else:
                b_source = self.var_arith_scalar.get()

            if op == "add":
                res = ops.image_add(img_a, b_source, mode=self.var_add_mode.get())
                math_desc = f"OPERASI PENJUMLAHAN:\nRumus: O(x,y) = A(x,y) + B(x,y)\nMode: {self.var_add_mode.get().upper()}"
            elif op == "sub":
                res = ops.image_subtract(img_a, b_source, absolute=self.var_sub_abs.get())
                math_desc = f"OPERASI PENGURANGAN:\nRumus: O(x,y) = {'|A - B|' if self.var_sub_abs.get() else 'clamp(A - B, 0, 255)'}"
            elif op == "mul":
                res = ops.image_multiply(img_a, b_source)
                math_desc = "OPERASI PERKALIAN:\nRumus: O(x,y) = (A · B) / 255"
            elif op == "div":
                res = ops.image_divide(img_a, b_source)
                math_desc = "OPERASI PEMBAGIAN:\nRumus: O(x,y) = (A / (B + 1)) · 255"
            elif op == "blend":
                alpha = self.var_blend_alpha.get()
                b_img = self.img_b_original if self.img_b_original is not None else self.img_a_original
                res = ops.image_blend(img_a, b_img, alpha=alpha)
                math_desc = f"OPERASI BLENDING (BOBOT):\nRumus: O = {alpha:.2f}·A + {1.0-alpha:.2f}·B"
            else:
                res = img_a

            self.img_result = res

        # TAB 2: Operasi Logika
        elif active_tab_idx == 2:
            op = self.var_logic_op.get()
            src_type = self.var_logic_source.get()

            if src_type == "shape_mask":
                mask = ops.create_shape_mask(
                    img_a.shape,
                    mask_type=self.var_mask_type.get(),
                    param=self.var_mask_size.get()
                )
                b_source = mask
            else:
                b_source = ops.resize_match(self.img_b_original if self.img_b_original is not None else self.img_a_original, img_a.shape)

            if op == "and":
                res = ops.boolean_and(img_a, b_source)
                math_desc = "OPERASI LOGIKA AND (∧):\nRumus: O(x,y) = A(x,y) & B(x,y)\nFungsi: Masking/Filtering piksel"
            elif op == "or":
                res = ops.boolean_or(img_a, b_source)
                math_desc = "OPERASI LOGIKA OR (∨):\nRumus: O(x,y) = A(x,y) | B(x,y)\nFungsi: Penggabungan citra"
            elif op == "not":
                res = ops.boolean_not(img_a)
                math_desc = "OPERASI LOGIKA NOT (¬):\nRumus: O(x,y) = ~A(x,y) = 255 - A(x,y)"
            elif op == "xor":
                res = ops.boolean_xor(img_a, b_source)
                math_desc = "OPERASI LOGIKA XOR (⊕):\nRumus: O(x,y) = A(x,y) ^ B(x,y)\nFungsi: Deteksi perbedaan/outline"
            elif op == "nand":
                res = ops.boolean_nand(img_a, b_source)
                math_desc = "OPERASI LOGIKA NAND:\nRumus: O(x,y) = ~(A & B)"
            elif op == "nor":
                res = ops.boolean_nor(img_a, b_source)
                math_desc = "OPERASI LOGIKA NOR:\nRumus: O(x,y) = ~(A | B)"
            else:
                res = img_a

            self.img_result = res

        self._render_canvases()
        self._update_matrix_display(math_desc)

    # -------------------------------------------------------------
    # CANVAS RENDERING & CLICK HANDLING
    # -------------------------------------------------------------
    def _render_canvases(self):
        if self.img_a_original is None:
            return

        # Render Canvas A
        self._draw_image_on_canvas(self.canvas_a, self.img_a_original)

        # Render Canvas B / Mask
        active_tab_idx = self.notebook.index(self.notebook.select())
        if active_tab_idx == 2 and self.var_logic_source.get() == "shape_mask":
            mask = ops.create_shape_mask(
                self.img_a_original.shape,
                mask_type=self.var_mask_type.get(),
                param=self.var_mask_size.get()
            )
            self._draw_image_on_canvas(self.canvas_b, mask)
        elif self.img_b_original is not None:
            matched_b = ops.resize_match(self.img_b_original, self.img_a_original.shape)
            self._draw_image_on_canvas(self.canvas_b, matched_b)
        else:
            self._draw_image_on_canvas(self.canvas_b, self.img_a_original)

        # Render Canvas Result
        if self.img_result is not None:
            self._draw_image_on_canvas(self.canvas_res, self.img_result)

    def _draw_image_on_canvas(self, canvas: tk.Canvas, img_arr: np.ndarray):
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 100)
        ch = max(canvas.winfo_height(), 100)

        h, w = img_arr.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        disp_w = max(1, int(w * scale))
        disp_h = max(1, int(h * scale))

        self.display_scale = scale

        # Resize PIL image for display
        pil_img = Image.fromarray(img_arr).resize((disp_w, disp_h), Image.Resampling.NEAREST)
        photo = ImageTk.PhotoImage(pil_img)

        canvas.delete("all")
        # Center in canvas
        ox = (cw - disp_w) // 2
        oy = (ch - disp_h) // 2
        canvas.create_image(ox, oy, image=photo, anchor="nw", tags="img")
        canvas.photo = photo
        canvas.offset = (ox, oy, disp_w, disp_h)

        # Draw selection marker if coordinate is set
        if self.selected_coord:
            sx, sy = self.selected_coord
            cx = ox + int(sx * scale)
            cy = oy + int(sy * scale)
            # Draw crosshair box
            canvas.create_rectangle(cx - 4, cy - 4, cx + 4, cy + 4, outline="#EF4444", width=2, tags="marker")
            canvas.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill="#FACC15", outline="#EF4444", tags="marker")

    def _on_canvas_click(self, event, source: str):
        if self.img_a_original is None:
            return

        canvas_map = {"A": self.canvas_a, "B": self.canvas_b, "Result": self.canvas_res}
        canvas = canvas_map[source]
        if not hasattr(canvas, "offset"):
            return

        ox, oy, disp_w, disp_h = canvas.offset
        rel_x = event.x - ox
        rel_y = event.y - oy

        if 0 <= rel_x < disp_w and 0 <= rel_y < disp_h:
            orig_x = int(rel_x / self.display_scale)
            orig_y = int(rel_y / self.display_scale)
            
            orig_w = self.img_a_original.shape[1]
            orig_h = self.img_a_original.shape[0]
            orig_x = min(max(0, orig_x), orig_w - 1)
            orig_y = min(max(0, orig_y), orig_h - 1)

            self.selected_coord = (orig_x, orig_y)
            self.last_clicked_source = source

            self._render_canvases()
            self._update_matrix_display()

    # -------------------------------------------------------------
    # MATRIX & PIXEL INSPECTOR UPDATE
    # -------------------------------------------------------------
    def _update_matrix_display(self, math_desc: str = None):
        if self.img_a_original is None or not self.selected_coord:
            return

        x, y = self.selected_coord
        source = self.last_clicked_source

        if source == "A":
            target_img = self.img_a_original
            src_name = "Citra Asli (A)"
        elif source == "B":
            active_tab_idx = self.notebook.index(self.notebook.select())
            if active_tab_idx == 2 and self.var_logic_source.get() == "shape_mask":
                target_img = ops.create_shape_mask(
                    self.img_a_original.shape,
                    mask_type=self.var_mask_type.get(),
                    param=self.var_mask_size.get()
                )
            else:
                target_img = ops.resize_match(self.img_b_original if self.img_b_original is not None else self.img_a_original, self.img_a_original.shape)
            src_name = "Citra B / Masker"
        else:
            target_img = self.img_result if self.img_result is not None else self.img_a_original
            src_name = "Citra Hasil Pemrosesan"

        # Extract Neighborhood Matrix
        size = self.var_matrix_size.get()
        info = ops.get_pixel_neighborhood(target_img, x, y, size=size)

        # Update Col 1: Detail Labels
        self.lbl_inspect_src.config(text=f"Sumber: {src_name}")
        self.lbl_inspect_coord.config(text=f"Koordinat (X, Y): ({x}, {y})")
        self.lbl_inspect_rgb.config(text=f"RGB: [{info['r']}, {info['g']}, {info['b']}]\nHEX: {info['hex']}")
        self.lbl_inspect_gray.config(text=f"Grayscale (Intensitas): {info['gray']}")
        self.lbl_inspect_bin.config(
            text=f"Biner 8-bit:\nR: {info['binary_r']}\nG: {info['binary_g']}\nB: {info['binary_b']}"
        )
        self.swatch_preview.config(bg=info['hex'])

        # Update Col 2: Matrix Visual Grid
        for widget in self.grid_container.winfo_children():
            widget.destroy()

        chan_sel = self.var_matrix_channel.get()
        if chan_sel == "Red (R)":
            mat = info["mat_r"]
            bg_base = "#FEE2E2"
        elif chan_sel == "Green (G)":
            mat = info["mat_g"]
            bg_base = "#DCFCE7"
        elif chan_sel == "Blue (B)":
            mat = info["mat_b"]
            bg_base = "#DBEAFE"
        else:  # Grayscale
            mat = info["mat_gray"]
            bg_base = "#F1F5F9"

        rows = len(mat)
        cols = len(mat[0]) if rows > 0 else 0
        cx = info["center_offset"]["cx"]
        cy = info["center_offset"]["cy"]

        for r_idx in range(rows):
            for c_idx in range(cols):
                val = mat[r_idx][c_idx]
                is_center = (r_idx == cy and c_idx == cx)
                
                cell_bg = "#FACC15" if is_center else bg_base
                cell_fg = "#000000" if is_center else "#1E293B"
                border_w = 2 if is_center else 1
                border_c = "#DC2626" if is_center else "#CBD5E1"

                cell_frame = tk.Frame(self.grid_container, bg=border_c, padx=border_w, pady=border_w)
                cell_frame.grid(row=r_idx, column=c_idx, padx=2, pady=2, sticky="nsew")

                cell_lbl = tk.Label(
                    cell_frame, text=str(val), bg=cell_bg, fg=cell_fg,
                    font=("Consolas", 10 if size == 3 else 8, "bold" if is_center else "normal"),
                    width=4 if size == 3 else 3, height=1
                )
                cell_lbl.pack(fill=tk.BOTH, expand=True)

        for i in range(cols):
            self.grid_container.columnconfigure(i, weight=1)
        for j in range(rows):
            self.grid_container.rowconfigure(j, weight=1)

        # Update Col 3: Mathematical Formula Details
        self.txt_math_calc.delete("1.0", tk.END)
        calc_text = f"📍 Titik ({x}, {y})\n"
        
        # Original A value at (x,y)
        p_a = self.img_a_original[y, x]
        if len(self.img_a_original.shape) == 3:
            gray_a = int(round(0.299 * p_a[0] + 0.587 * p_a[1] + 0.114 * p_a[2]))
            calc_text += f"Citra A: RGB [{p_a[0]}, {p_a[1]}, {p_a[2]}] (Gray: {gray_a})\n"
        else:
            calc_text += f"Citra A: Gray={p_a}\n"

        # Result value at (x,y)
        if self.img_result is not None:
            p_res = self.img_result[y, x]
            if len(self.img_result.shape) == 3:
                gray_res = int(round(0.299 * p_res[0] + 0.587 * p_res[1] + 0.114 * p_res[2]))
                calc_text += f"Citra Hasil: RGB [{p_res[0]}, {p_res[1]}, {p_res[2]}]\n"
                calc_text += f"Citra Hasil (Grayscale): {gray_res}\n"
            else:
                calc_text += f"Citra Hasil: Gray={p_res}\n"

        calc_text += "\n" + "-"*28 + "\n"
        active_tab_idx = self.notebook.index(self.notebook.select())
        if active_tab_idx == 0:
            b_val = self.var_brightness.get()
            c_val = self.var_contrast.get()
            g_val = self.var_gamma.get()
            calc_text += "Formula Operasi Piksel:\n"
            calc_text += f"• Pencerahan: min(I + ({b_val}), 255)\n"
            calc_text += f"• Kontras: 128 + {c_val:.2f}*(I - 128)\n"
            calc_text += f"• Gamma: 255 * (I/255)^(1/{g_val:.2f})\n"
            calc_text += "• Invert: 255 - I\n"
        elif active_tab_idx == 1:
            op = self.var_arith_op.get()
            target_type = self.var_arith_target.get()
            val_b = self.var_arith_scalar.get() if target_type == "scalar" else "Citra B"
            calc_text += "Formula Aritmatika:\n"
            if op == "add":
                calc_text += f"O(x,y) = A(x,y) + {val_b}\n(Clamped ke [0..255])\n"
            elif op == "sub":
                calc_text += f"O(x,y) = A(x,y) - {val_b}\n"
            elif op == "mul":
                calc_text += f"O(x,y) = (A(x,y) * {val_b}) / 255\n"
            elif op == "div":
                calc_text += f"O(x,y) = (A(x,y) / ({val_b} + 1)) * 255\n"
            elif op == "blend":
                al = self.var_blend_alpha.get()
                calc_text += f"O(x,y) = {al:.2f}*A + {1.0-al:.2f}*B\n"
        elif active_tab_idx == 2:
            op = self.var_logic_op.get().upper()
            calc_text += "Formula Logika Boolean:\n"
            calc_text += f"O(x,y) = A(x,y) {op} B_Mask(x,y)\n"
            calc_text += "(Operasi dilakukan bit demi bit 8-bit)\n"

        self.txt_math_calc.insert(tk.END, calc_text)


def main():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
