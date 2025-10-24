# YOLOv8-CropDensity

**Automated Agricultural Plot Segmentation and Vegetation Density Analysis Using YOLOv8 Nano**

---

## Project Overview
This project implements an **automated pipeline** for:  

1. **Segmenting agricultural plots** from RGB images using **YOLOv8 Nano**.  
2. **Estimating vegetation density** via a machine learning-based approach.  
3. **Visualizing low- and high-density areas** using transparent overlays (red = low, green = high).  

**Applications:** Precision agriculture, crop health monitoring, identifying underperforming plots, and supporting actionable farm management.

---

## Dataset
- **Name:** seg-parcel v1  
- **Source:** [Roboflow](https://roboflow.com)  
- **Images:** 413 images annotated in YOLOv8 format  
- **Preprocessing:** Resize to 640×640, auto-contrast, orientation correction  
- **Augmentation:** Horizontal flip, 90° rotations, random cropping, salt-and-pepper noise  

*Sample images with segmentation masks can be found in the `results/` folder.*

---

## Installation

```bash
# Clone the repository
git clone https://github.com/jeeva-m-21/YOLOv8-CropDensity.git
cd YOLOv8-CropDensity

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install ultralytics opencv-python matplotlib numpy
````

---

## Usage

1. **Load the model and run segmentation:**

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load YOLOv8 Nano segmentation model
model = YOLO("weights/best.pt")

# Predict segmentation on an image
results = model.predict("data/sample_image.jpg")

# Visualize the segmented image
seg_image = results[0].plot()
seg_image_rgb = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
plt.imshow(seg_image_rgb)
plt.axis('off')
plt.show()
```

2. **Vegetation Density Analysis:**

* Extract features (color, texture) from segmented plots.
* Use the trained ML model to predict vegetation density.
* Overlay **low-density (red)** and **high-density (green)** regions for visualization.

---

## Results

**Validation Metrics (YOLOv8 Nano Segmentation):**

| Class            | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
| ---------------- | ----- | ----- | --------- | ------------ | ------ | ------ | ---------- | ------------- |
| Non-uniform plot | 0.744 | 0.891 | 0.921     | 0.778        | 0.744  | 0.891  | 0.921      | 0.836         |
| Uniform plot     | 0.773 | 0.787 | 0.869     | 0.748        | 0.773  | 0.787  | 0.869      | 0.813         |
| **All plots**    | 0.759 | 0.839 | 0.895     | 0.763        | 0.759  | 0.839  | 0.895      | 0.825         |

* **Inference Speed:** ~0.3 ms preprocessing, 2.8 ms inference, 7.4 ms postprocessing per image
* **Model Size:** 3.26M parameters, 11.3 GFLOPs

*Interpretation:* The model performs slightly better on non-uniform plots and achieves **high segmentation accuracy**, while maintaining lightweight and fast inference suitable for real-time use.

*Vegetation Density Visualization:*

* Low-density areas highlighted in **red**, high-density in **green**.
* Provides actionable insights for crop health monitoring.

---

## Folder Structure

```
YOLOv8-CropDensity/
│
├── data/         # Dataset
├── weights/      # YOLOv8 Nano trained weights
├── scripts/      # Segmentation & vegetation density scripts
├── results/      # Segmented images and overlay outputs
├── README.md
└── requirements.txt
```

---

## Future Work

* Integrate multispectral images for NDVI-based vegetation analysis
* Improve ML-based vegetation density prediction using CNNs
* Deploy as a web/mobile application for real-time agricultural monitoring

---

## License

MIT License

