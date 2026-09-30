"""
image_operations.py
Modul inti pemrosesan citra digital:
1. Operasi Piksel (Aras Titik)
2. Operasi Aritmatika Citra
3. Operasi Logika (Boolean) Citra
4. Helper Inspeksi Nilai Matriks Piksel
"""

import numpy as np
import cv2

# ==========================================
# 1. OPERASI PIKSEL (ARAS TITIK)
# ==========================================

def adjust_brightness(img: np.ndarray, value: int) -> np.ndarray:
    """
    Rumus: O(x,y) = clamp(I(x,y) + b, 0, 255)
    """
    res = img.astype(np.int16) + value
    return np.clip(res, 0, 255).astype(np.uint8)

def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Rumus: O(x,y) = clamp(c * I(x,y), 0, 255)
    atau kontras terhadap titik tengah: O(x,y) = clamp(128 + factor * (I(x,y) - 128), 0, 255)
    """
    res = 128.0 + factor * (img.astype(np.float32) - 128.0)
    return np.clip(res, 0, 255).astype(np.uint8)

def invert_image(img: np.ndarray) -> np.ndarray:
    """
    Rumus: O(x,y) = 255 - I(x,y) (Citra Negatif)
    """
    return 255 - img

def threshold_image(img: np.ndarray, threshold: int) -> np.ndarray:
    """
    Rumus: O(x,y) = 255 jika I_gray(x,y) >= T else 0
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    if len(img.shape) == 3:
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return binary

def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Rumus: O(x,y) = 255 * (I(x,y) / 255) ^ (1 / gamma)
    """
    inv_gamma = 1.0 / (gamma if gamma > 0 else 0.0001)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def log_transform(img: np.ndarray) -> np.ndarray:
    """
    Rumus: O(x,y) = c * log(1 + I(x,y)) dengan c = 255 / log(1 + max_val)
    """
    img_float = img.astype(np.float32)
    c = 255.0 / np.log(1.0 + np.max(img_float) if np.max(img_float) > 0 else 1.0)
    log_img = c * np.log(1.0 + img_float)
    return np.clip(log_img, 0, 255).astype(np.uint8)

def to_grayscale(img: np.ndarray, method: str = "luminance") -> np.ndarray:
    """
    Metode:
    - 'luminance': 0.299*R + 0.587*G + 0.114*B
    - 'average': (R + G + B) / 3
    - 'red_only': R channel
    - 'green_only': G channel
    - 'blue_only': B channel
    """
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)

    if method == "luminance":
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    elif method == "average":
        gray = (r + g + b) / 3.0
    elif method == "red_only":
        gray = r
    elif method == "green_only":
        gray = g
    elif method == "blue_only":
        gray = b
    else:
        gray = 0.299 * r + 0.587 * g + 0.114 * b

    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2RGB)


# ==========================================
# 2. OPERASI ARITMATIKA CITRA
# ==========================================

def resize_match(img_b: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Menyamakan resolusi Citra B agar sama persis dengan Citra A."""
    h, w = target_shape[:2]
    if img_b.shape[:2] != (h, w):
        img_b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_LINEAR)
    if len(target_shape) == 3 and len(img_b.shape) == 2:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2RGB)
    elif len(target_shape) == 2 and len(img_b.shape) == 3:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    return img_b

def image_add(img_a: np.ndarray, img_b_or_scalar, mode="clamp") -> np.ndarray:
    """
    Penjumlahan Citra: A + B atau A + skalar
    Mode:
    - 'clamp' (saturasi standar): min(A + B, 255)
    - 'modulo': (A + B) % 256
    """
    if isinstance(img_b_or_scalar, np.ndarray):
        b = resize_match(img_b_or_scalar, img_a.shape)
    else:
        b = img_b_or_scalar

    if mode == "modulo":
        res = (img_a.astype(np.int32) + b) % 256
        return res.astype(np.uint8)
    else:
        res = img_a.astype(np.int32) + b
        return np.clip(res, 0, 255).astype(np.uint8)

def image_subtract(img_a: np.ndarray, img_b_or_scalar, absolute: bool = False) -> np.ndarray:
    """
    Pengurangan Citra: A - B atau |A - B|
    """
    if isinstance(img_b_or_scalar, np.ndarray):
        b = resize_match(img_b_or_scalar, img_a.shape)
    else:
        b = img_b_or_scalar

    res = img_a.astype(np.int32) - b
    if absolute:
        res = np.abs(res)
    return np.clip(res, 0, 255).astype(np.uint8)

def image_multiply(img_a: np.ndarray, img_b_or_scalar, normalize: bool = True) -> np.ndarray:
    """
    Perkalian Citra: (A * B) / 255 atau A * scalar
    """
    if isinstance(img_b_or_scalar, np.ndarray):
        b = resize_match(img_b_or_scalar, img_a.shape).astype(np.float32)
        if normalize:
            res = (img_a.astype(np.float32) * b) / 255.0
        else:
            res = img_a.astype(np.float32) * b
    else:
        res = img_a.astype(np.float32) * float(img_b_or_scalar)
    return np.clip(res, 0, 255).astype(np.uint8)

def image_divide(img_a: np.ndarray, img_b_or_scalar, scale_result: bool = True) -> np.ndarray:
    """
    Pembagian Citra: (A / (B + 1)) * 255 atau A / scalar
    """
    if isinstance(img_b_or_scalar, np.ndarray):
        b = resize_match(img_b_or_scalar, img_a.shape).astype(np.float32)
        # Avoid division by zero
        b = np.where(b == 0, 1.0, b)
        if scale_result:
            res = (img_a.astype(np.float32) / b) * 255.0
        else:
            res = img_a.astype(np.float32) / b
    else:
        val = float(img_b_or_scalar) if float(img_b_or_scalar) != 0 else 1.0
        res = img_a.astype(np.float32) / val
    return np.clip(res, 0, 255).astype(np.uint8)

def image_blend(img_a: np.ndarray, img_b: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blending Citra: O = alpha * A + (1 - alpha) * B
    """
    b = resize_match(img_b, img_a.shape)
    alpha = max(0.0, min(1.0, alpha))
    res = alpha * img_a.astype(np.float32) + (1.0 - alpha) * b.astype(np.float32)
    return np.clip(res, 0, 255).astype(np.uint8)


# ==========================================
# 3. OPERASI LOGIKA (BOOLEAN / BITWISE)
# ==========================================

def boolean_and(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Operasi Logika Bitwise AND: A AND B"""
    b = resize_match(img_b, img_a.shape)
    return cv2.bitwise_and(img_a, b)

def boolean_or(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Operasi Logika Bitwise OR: A OR B"""
    b = resize_match(img_b, img_a.shape)
    return cv2.bitwise_or(img_a, b)

def boolean_not(img_a: np.ndarray) -> np.ndarray:
    """Operasi Logika Bitwise NOT: ~A"""
    return cv2.bitwise_not(img_a)

def boolean_xor(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Operasi Logika Bitwise XOR: A XOR B"""
    b = resize_match(img_b, img_a.shape)
    return cv2.bitwise_xor(img_a, b)

def boolean_nand(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Operasi Logika Bitwise NAND: NOT (A AND B)"""
    and_res = boolean_and(img_a, img_b)
    return cv2.bitwise_not(and_res)

def boolean_nor(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Operasi Logika Bitwise NOR: NOT (A OR B)"""
    or_res = boolean_or(img_a, img_b)
    return cv2.bitwise_not(or_res)

def create_shape_mask(shape: tuple, mask_type: str = "circle", param: int = 100) -> np.ndarray:
    """
    Membuat masker biner geometris:
    - 'circle': Lingkaran di tengah
    - 'rect': Kotak di tengah
    - 'diamond': Belah ketupat di tengah
    - 'checker': Papan catur
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2

    if mask_type == "circle":
        r = min(param, min(h, w) // 2)
        cv2.circle(mask, (cx, cy), r, 255, -1)
    elif mask_type == "rect":
        sz = min(param, min(h, w) // 2)
        cv2.rectangle(mask, (max(0, cx - sz), max(0, cy - sz)), (min(w, cx + sz), min(h, cy + sz)), 255, -1)
    elif mask_type == "diamond":
        sz = min(param, min(h, w) // 2)
        pts = np.array([[cx, cy - sz], [cx + sz, cy], [cx, cy + sz], [cx - sz, cy]], np.int32)
        cv2.fillPoly(mask, [pts], 255)
    elif mask_type == "checker":
        block_size = max(10, param // 2)
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if (x // block_size + y // block_size) % 2 == 0:
                    mask[y:min(y+block_size, h), x:min(x+block_size, w)] = 255

    if len(shape) == 3:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return mask


# ==========================================
# 4. HELPER INSPEKSI MATRIKS PIKSEL
# ==========================================

def get_pixel_neighborhood(img: np.ndarray, x: int, y: int, size: int = 3, channel: str = "RGB") -> dict:
    """
    Mengambil sub-matriks di sekitar koordinat (x, y) berukuran size x size (e.g. 3x3, 5x5).
    Mengembalikan dict berisi metadata piksel, matriks tetangga, dan rincian channel.
    """
    h, w = img.shape[:2]
    # Batasi koordinat dalam rentang gambar
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    
    half = size // 2
    y_min = max(0, y - half)
    y_max = min(h, y + half + 1)
    x_min = max(0, x - half)
    x_max = min(w, x + half + 1)

    # Sub array
    sub_img = img[y_min:y_max, x_min:x_max]

    # Pixel center value
    pixel_val = img[y, x]
    if len(img.shape) == 3:
        r, g, b = int(pixel_val[0]), int(pixel_val[1]), int(pixel_val[2])
        gray_val = int(0.299 * r + 0.587 * g + 0.114 * b)
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        binary_r = f"{r:08b}"
        binary_g = f"{g:08b}"
        binary_b = f"{b:08b}"
    else:
        gray_val = int(pixel_val)
        r, g, b = gray_val, gray_val, gray_val
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        binary_r = binary_g = binary_b = f"{gray_val:08b}"

    # Matrix for specific view
    if len(img.shape) == 3:
        mat_r = sub_img[:, :, 0]
        mat_g = sub_img[:, :, 1]
        mat_b = sub_img[:, :, 2]
        mat_gray = (0.299 * mat_r + 0.587 * mat_g + 0.114 * mat_b).astype(np.uint8)
    else:
        mat_r = mat_g = mat_b = mat_gray = sub_img

    return {
        "x": x,
        "y": y,
        "r": r,
        "g": g,
        "b": b,
        "gray": gray_val,
        "hex": hex_code,
        "binary_r": binary_r,
        "binary_g": binary_g,
        "binary_b": binary_b,
        "size": size,
        "bounds": {"x_min": x_min, "x_max": x_max - 1, "y_min": y_min, "y_max": y_max - 1},
        "center_offset": {"cx": x - x_min, "cy": y - y_min},
        "mat_r": mat_r.tolist(),
        "mat_g": mat_g.tolist(),
        "mat_b": mat_b.tolist(),
        "mat_gray": mat_gray.tolist(),
    }
