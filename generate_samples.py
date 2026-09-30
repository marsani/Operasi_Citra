# pyrefly: ignore [missing-import]
import cv2
import numpy as np

# Generate sample A: A rich colorful test image (shapes, gradients, text)
img_a = np.zeros((300, 300, 3), dtype=np.uint8)
# Gradient background
for y in range(300):
    for x in range(300):
        img_a[y, x] = [int(x / 300 * 255), int(y / 300 * 200), 180]

# Add shapes
cv2.circle(img_a, (150, 150), 70, (255, 255, 0), -1) # Cyan/Yellow in BGR
cv2.rectangle(img_a, (50, 50), (120, 120), (0, 0, 255), -1) # Red
cv2.putText(img_a, "CITRA A", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
cv2.imwrite("samples/sample_a.jpg", img_a)

# Generate sample B: Another image for arithmetic & boolean operations
img_b = np.zeros((300, 300, 3), dtype=np.uint8)
for y in range(300):
    for x in range(300):
        img_b[y, x] = [int((300-x)/300 * 220), int((300-y)/300 * 220), int((x+y)/600 * 255)]

cv2.rectangle(img_b, (130, 80), (250, 200), (0, 255, 0), -1) # Green
cv2.circle(img_b, (90, 200), 50, (255, 0, 255), -1)
cv2.putText(img_b, "CITRA B", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
cv2.imwrite("samples/sample_b.jpg", img_b)

# Generate Binary Mask
mask = np.zeros((300, 300), dtype=np.uint8)
cv2.circle(mask, (150, 150), 100, 255, -1)
cv2.imwrite("samples/sample_mask.png", mask)

print("Sample images generated successfully!")
