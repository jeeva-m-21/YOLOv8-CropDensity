# ---------------- 1️⃣ Install dependencies ----------------
# pip install ultralytics opencv-python matplotlib numpy

# ---------------- 2️⃣ Import libraries ----------------
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ---------------- 3️⃣ Load your trained model ----------------
model_path = r".\resources\weights\best.pt"
model = YOLO(model_path)

# ---------------- 4️⃣ Open file dialog to choose image ----------------
Tk().withdraw()
img_path = askopenfilename(title="Select an image to segment",
                           filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])

if not img_path:
    print("No image selected, exiting...")
    exit()

# ---------------- 5️⃣ Read image ----------------
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------------- 6️⃣ Run segmentation ----------------
results = model.predict(img_path, verbose=False)

# ---------------- 7️⃣ Process each detected plot ----------------
# We'll create a transparent overlay for vegetation density
overlay = img.copy()

for r in results[0].masks.data:  # each mask for segmented plot
    # r shape: (H, W) binary mask
    mask = r.cpu().numpy().astype(np.uint8)  # 0/1 mask
    mask_3ch = cv2.merge([mask]*3)  # convert to 3-channel

    # Extract the segmented plot area
    plot_area = cv2.bitwise_and(img, img, mask=mask)

    # Convert plot area to grayscale for vegetation density approximation
    plot_gray = cv2.cvtColor(plot_area, cv2.COLOR_BGR2GRAY)

    # Normalize to get vegetation "density"
    density = cv2.normalize(plot_gray, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold for low/high density (tunable)
    low_density_mask = (density < 100).astype(np.uint8)
    high_density_mask = (density > 150).astype(np.uint8)

    # Color overlays: green for high, red for low density
    overlay[np.where(low_density_mask)] = (0, 0, 255)       # Red for low
    overlay[np.where(high_density_mask)] = (0, 255, 0)      # Green for high

# Blend overlay with original image (transparent effect)
alpha = 0.4
result_img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

# ---------------- 8️⃣ Display final image ----------------
plt.figure(figsize=(12,12))
plt.imshow(result_rgb)
plt.axis('off')
plt.show()
