

# YOLOv8-CropDensity

**Automated Agricultural Plot Segmentation and Vegetation Density Analysis Using YOLOv8 Nano**


## **Project Overview**

This project implements an **automated pipeline** to:

1. Segment agricultural plots using **YOLOv8 Nano**.
2. Analyze vegetation density using a **machine learning–based approach**.
3. Visualize **low-density (red)** and **high-density (green)** areas with transparent overlays.

**Applications:** Precision agriculture, crop health monitoring, and identifying underperforming plots.

---

## **Dataset**

* **Name:** seg-parcel v1
* **Source:** [Roboflow](https://roboflow.com)
* **Images:** 413 annotated images (YOLOv8 format)
* **Preprocessing:** Resize to 640×640, auto-contrast, orientation correction
* **Augmentation:** Horizontal flip, rotations, random crop, salt-and-pepper noise

---

## **Evaluation Metrics**

**Segmentation:**

* mAP50, mAP50-95, IoU, Precision, Recall, F1-score

**Vegetation Density:**

* Regression: MSE, RMSE, R²
* Classification: Accuracy, F1-score

**Inference Speed:** ~3–10ms per image (YOLOv8 Nano)

---

## **Results**

* Segmentation accuracy: **mAP50 ~ 0.895**
* Mask precision/recall: **0.75–0.84**
* YOLOv8 Nano is fast and lightweight (~3.2M parameters)
* Transparent overlays highlight low/high vegetation areas clearly

*Include images of segmented plots and vegetation overlays in the `results/` folder.*

---


## **Future Work**

* Integrate multispectral images for NDVI-based vegetation density analysis
* Improve ML-based vegetation density estimation using CNNs
* Deploy as a web or mobile application for real-time field monitoring

---

## **License**

MIT License

---

