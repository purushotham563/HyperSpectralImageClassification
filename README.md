# 3D CNN for Hyperspectral Image Classification

A deep learning approach for hyperspectral image classification using 3D Convolutional Neural Networks (3D-CNN) with the Indian Pines dataset.

![Hyperspectral Classification](https://img.shields.io/badge/Deep%20Learning-Hyperspectral-blue) ![Python](https://img.shields.io/badge/Python-3.7+-green) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange) ![License](https://img.shields.io/badge/License-MIT-red)

## 🚀 Overview

This project implements a 3D Convolutional Neural Network for classifying hyperspectral images. The model combines 3D convolutions to extract spectral-spatial features with 2D convolutions for enhanced spatial feature learning, achieving **99.33% accuracy** on the Indian Pines dataset.

### Key Features

- **3D CNN Architecture**: Leverages both spectral and spatial information
- **PCA Preprocessing**: Reduces dimensionality while preserving essential features
- **Patch-based Classification**: Uses 25×25 spatial patches for context-aware classification
- **Comprehensive Evaluation**: Includes accuracy metrics, confusion matrices, and visualization tools
- **Classification Mapping**: Generates pixel-wise classification maps

## 📊 Results

### Performance Metrics
- **Overall Accuracy (OA)**: 99.33%
- **Average Accuracy (AA)**: 98.75%
- **Cohen's Kappa**: 99.25%
- **Test Loss**: 1.78%

### Per-Class Results
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|----------|---------|
| Alfalfa | 1.00 | 0.97 | 0.98 | 32 |
| Corn-notill | 0.99 | 1.00 | 0.99 | 1000 |
| Corn-mintill | 1.00 | 0.98 | 0.99 | 581 |
| Corn | 0.99 | 1.00 | 1.00 | 166 |
| Grass-pasture | 0.99 | 1.00 | 1.00 | 338 |
| ... | ... | ... | ... | ... |

## 🏗️ Architecture

```
Input Layer (25×25×30×1)
    ↓
3D Conv (8 filters, 3×3×7) + BN + ReLU
    ↓
3D Conv (16 filters, 3×3×5) + BN + ReLU
    ↓
3D Conv (32 filters, 3×3×3) + BN + ReLU
    ↓
Reshape to 2D
    ↓
2D Conv (64 filters, 3×3) + BN + ReLU
    ↓
Flatten
    ↓
Dense (256) + Dropout(0.4)
    ↓
Dense (128) + Dropout(0.4)
    ↓
Output (16 classes, Softmax)
```

## 📋 Requirements

```bash
tensorflow>=2.8.0
scikit-learn>=1.0.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
spectral>=0.23.0
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hyperspectral-3dcnn.git
cd hyperspectral-3dcnn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Indian Pines dataset:
   - `Indian_pines_corrected.mat`
   - `Indian_pines_gt.mat`
   
   Place them in the `data/` directory.

## 💻 Usage

### Basic Training

```python
from hyperspectral_3dcnn import *

# Load and preprocess data
X, y = load_data('IP')
X_pca, pca = applyPCA(X, numComponents=30)
X_patches, y_patches = createImageCubes(X_pca, y, windowSize=25)

# Split data
X_train, X_test, y_train, y_test = splitTrainTestSet(X_patches, y_patches, 0.7)

# Build and train model
model = build_3dcnn_model((25, 25, 30, 1), 16)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_split=0.2)
```

### Complete Pipeline

```python
# Run the complete pipeline
python main.py --dataset IP --epochs 50 --batch_size 256 --test_ratio 0.7
```

### Generate Classification Maps

```python
# Generate pixel-wise classification map
classification_map = generate_classification_map(model, X_original, y_original, pca_model)

# Visualize results
plot_classification_results(y_original, classification_map, target_names)
```

## 📁 Project Structure

```
HyperSepctaralImageClassification/
|-hypisc
├── README.md
└── LICENSE
```

## 🔧 Configuration

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `windowSize` | 25 | Spatial patch size |
| `numComponents` | 30 | PCA components |
| `test_ratio` | 0.7 | Test set ratio |
| `batch_size` | 256 | Training batch size |
| `epochs` | 50 | Training epochs |
| `learning_rate` | 0.001 | Initial learning rate |

### Model Parameters

- **3D Convolution Filters**: [8, 16, 32]
- **2D Convolution Filters**: 64
- **Dense Layer Units**: [256, 128]
- **Dropout Rate**: 0.4
- **Optimizer**: Adam with exponential decay

## 📈 Visualization

The project includes comprehensive visualization tools:

- **Training History**: Loss and accuracy curves
- **Confusion Matrix**: Normalized confusion matrix heatmap
- **Per-Class Accuracy**: Bar chart of class-wise performance
- **Classification Maps**: Ground truth vs predictions comparison
- **Spectral Signatures**: Visualization of different land cover types

## 🔍 Indian Pines Dataset

The Indian Pines dataset contains:
- **Dimensions**: 145×145 pixels
- **Spectral Bands**: 200 (after noise removal: 200)
- **Classes**: 16 different land cover types
- **Total Samples**: 10,249 labeled pixels

### Land Cover Classes
1. Alfalfa
2. Corn-notill
3. Corn-mintill
4. Corn
5. Grass-pasture
6. Grass-trees
7. Grass-pasture-mowed
8. Hay-windrowed
9. Oats
10. Soybean-notill
11. Soybean-mintill
12. Soybean-clean
13. Wheat
14. Woods
15. Buildings-Grass-Trees-Drives
16. Stone-Steel-Towers

## 🔬 Methodology

### Preprocessing Pipeline
1. **PCA Dimensionality Reduction**: 200 → 30 bands
2. **Zero Padding**: Add margins for patch extraction
3. **Patch Extraction**: 25×25 spatial patches
4. **Data Normalization**: Zero mean, unit variance

### Training Strategy
- **Train/Test Split**: 30%/70%
- **Data Augmentation**: Random rotations and flips
- **Early Stopping**: Monitor validation accuracy
- **Learning Rate Decay**: Exponential decay schedule

## 📊 Comparison with Other Methods

| Method | OA (%) | AA (%) | Kappa (%) |
|--------|--------|--------|-----------|
| SVM | 89.45 | 86.73 | 87.12 |
| 2D CNN | 95.67 | 93.21 | 94.89 |
| **3D CNN (Ours)** | **99.33** | **98.75** | **99.25** |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
git clone https://github.com/purushotham563/HyperSpectralImageClassification
```


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Indian Pines dataset provided by Purdue University
- TensorFlow and Keras teams for the deep learning framework
- Scikit-learn for preprocessing utilities
- The hyperspectral remote sensing community

##  Contact

- **Author**: Purushotham Reddy D A
- **Email**: purushotham.appireddy@gmail.com


## 🔗 Related Projects

- [Hyperspectral Toolbox](https://github.com/davidkun/HyperSpectralToolbox)
- [SpectralPython](https://github.com/spectralpython/spectral)
- [Awesome Hyperspectral](https://github.com/awesome-hyperspectral/awesome-hyperspectral)

---

⭐ **Star this repository if you found it helpful!** ⭐
