# YOLOv8-MAH

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![GitHub stars](https://img.shields.io/github/stars/KardeniaPoyu/YOLOv8-MAH?style=social)

**YOLOv8-MAH: Vehicle Detection with Mosaic Augmentation, Attention Mechanism & Heatmap Interpretability**

---

## Table of Contents

- [📖 Abstract](#-abstract)
- [🚀 Features](#-features)
- [📊 Performance](#-performance)
- [📈 Usage](#-usage)
- [📂 Project Structure](#-project-structure)
- [📷 Visualization](#-visualization)
- [🛠 Installation](#-installation)
- [📜 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)
- [📬 Contact](#-contact)

---

## 📖 Abstract

**YOLOv8-MAH** enhances YOLOv8 with:

- **Adaptive Mosaic Augmentation** – improves training data diversity.  
- **Attention-Guided Feature Enhancement Module** – boosts feature extraction for better detection.  
- **Heatmap-Based Interpretability** – visual insights into model decisions.  

Comparative experiments with **YOLOv5–YOLOv11** show:

- **mAP50:** 0.92932  
- **mAP50-95:** 0.71775  
- **Precision:** 0.89915  

> YOLOv8-MAH achieves superior accuracy, robustness, and interpretability for intelligent transportation systems.

**Keywords:** YOLOv8-MAH, Vehicle Detection, Mosaic Augmentation, Attention, Heatmap Interpretability

---

## 🚀 Features

- **High Accuracy:** Outperforms baseline YOLO models.  
- **Robust Generalization:** Handles diverse scenarios.  
- **Interpretable Results:** Heatmap visualization for model decisions.  
- **Efficient Training:** Low training & validation losses.  

---

## 📊 Performance

| Model        | mAP50   | mAP50-95 | Precision | Recall  | Train Loss | Val Loss |
|-------------|---------|-----------|-----------|--------|------------|----------|
| YOLOv5      | 0.92151 | 0.71058   | 0.90403   | 0.87765 | 0.66518    | 0.95083 |
| YOLOv8      | 0.92425 | 0.72112   | 0.90221   | 0.86595 | 0.62414    | 0.94234 |
| YOLOv11     | 0.92347 | 0.60071   | 0.91603   | 0.87700 | 0.89455    | 1.37169 |
| YOLOv9      | 0.92220 | 0.70664   | 0.91491   | 0.87710 | 0.62963    | 0.97373 |
| YOLOv10     | 0.90593 | 0.68250   | 0.89035   | 0.84204 | 1.34783    | 2.00352 |
| **YOLOv8-MAH** | 0.92932 | 0.71775   | 0.89915   | 0.89767 | 0.67916    | 0.94080 |

By comparing the results of different YOLO models, it is observed that **YOLOv8-MAH** achieves the best detection accuracy and robustness. The training dynamics, including loss convergence and evaluation metrics over epochs, are visualized in the `docs/training_curve.gif`.

---

## 📈 Usage

### Training

```bash
python train.py --data data.yaml --weights yolov8-mah.pt --epochs 100
```

### Inference

```bash
python detect.py --weights yolov8-mah.pt --source path/to/data
```

### Generate Heatmaps

```bash
python visualize_heatmap.py --weights yolov8-mah.pt --image path/to/image
```

---

## 📂 Project Structure

```
YOLOv8-MAH/
├── data/                 # Dataset and configuration files
├── models/               # Model architectures and pretrained weights
├── utils/                # Utility scripts for augmentation & visualization
├── train.py              # Training script
├── detect.py             # Inference script
├── visualize_heatmap.py  # Heatmap visualization
├── docs/                 # Visualization images, GIFs, diagrams
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 📷 Visualization

**Example Detection Results**  
![Vehicle Detection](docs/vehicle_example.jpg)

**Heatmap Interpretability**  
![Heatmap](docs/heatmap_example.jpg)

**Training Curve GIF**  
![Training GIF](docs/training_curve.gif)

**Model Structure Diagram**  
![YOLOv8-MAH Architecture](docs/model_structure.png)

> Place your images/GIFs in `docs/` for proper rendering.

---

## 🛠 Installation

```bash
git clone https://github.com/KardeniaPoyu/YOLOv8-MAH.git
cd YOLOv8-MAH
pip install -r requirements.txt
```

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE).

---

## 🙌 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- [PyTorch](https://pytorch.org/)  
- Open-source community contributors

---

## 📬 Contact

For questions or contributions, please open an issue or contact: **yirong.zhou@muc.edu.cn**
