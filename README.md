# Brain Tumor Classification Using Deep Learning

## Overview
This project aims to develop a deep learning-based system for classifying brain tumors using MRI images. The classification categories include:
- Glioma
- Meningioma
- Pituitary tumor
- No tumor

The system leverages convolutional neural networks (CNNs) to achieve high accuracy and integrates Grad-CAM visualizations for interpretability. The ultimate goal is to aid medical professionals by reducing diagnostic workload and improving efficiency and precision in treatment planning.

## Motivation
Brain tumors can pose significant health risks, requiring timely and accurate diagnosis. MRI imaging provides detailed insights into brain structures, but manual interpretation can be time-consuming and error-prone. This project addresses these challenges by automating tumor classification with advanced machine learning techniques.

## Dataset
The dataset consists of 7,023 MRI images obtained from two sources:

1. **Figshare Dataset**
   - Includes 3,064 high-resolution MRI images categorized into glioma, meningioma, and pituitary tumor classes.
   - Highlights specific regions affected by tumors.

2. **BR35H Dataset**
   - Comprises MRI scans categorized into two classes: tumor and no tumor.
   - Provides additional images for the "no tumor" category.

The dataset is curated to include the following classes:
- Glioma
- Meningioma
- Pituitary tumor
- No tumor

**Dataset Access**: The dataset can be downloaded from a [Google Drive link](#).

## Models
Several state-of-the-art CNN architectures were used in this project:

- **ResNet**: ResNet50 and ResNet152 for addressing the vanishing gradient problem.
- **VGGNet**: VGG16 and VGG19 for their simplicity and high accuracy.
- **DenseNet**: DenseNet121 for efficient parameter usage and feature reuse.
- **EfficientNet**: EfficientNetB0 for lightweight real-time applications.
- **Inception**: InceptionV3 for multi-scale feature extraction.
- **Custom CNN**: A lightweight model built using PyTorch for baseline performance.

### Grad-CAM Visualization
Grad-CAM (Gradient-weighted Class Activation Mapping) is employed to provide interpretability by highlighting the regions of MRI images that contribute most to the model's predictions.

## Evaluation Metrics
The models were evaluated using the following metrics:

1. **Accuracy**
2. **Precision**
3. **Recall**
4. **F1-Score**
5. **ROC-AUC**

## Results
- VGG16 achieved the highest accuracy of 98%, with sharp and localized Grad-CAM visualizations.
- ResNet50, DenseNet121, and InceptionV3 also demonstrated strong performance.
- Custom CNN achieved 90% accuracy but lacked robustness compared to pretrained models.

## Dataset Preprocessing
1. **Splitting**: The dataset is divided into training (70%), validation (15%), and testing (15%).
2. **Normalization**: All pixel values are rescaled to [0, 1] by dividing by 255.0.
3. **Resizing**: Images are resized to 128x128 for uniformity.
4. **Batching**: Images are processed in batches of 32 for efficient training.

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification.git
   ```
2. Download the dataset from the provided [Google Drive link](#).
3. Run the training script:
   ```bash
   jupyter notebook brain_tumor_classification.ipynb
   ```

## Future Work
- Explore ensemble models to improve generalization.
- Train on larger and more diverse datasets for better clinical applicability.
- Optimize models for deployment on mobile and edge devices.

## References
- Hussain, S., et al. "Brain Tumor Segmentation Using 3D Convolutional Neural Networks."
- Rezaei, T., et al. "Deep Learning for Brain Tumor Segmentation and Classification in MRI Images."
- Pereira, P., et al. "Automated Classification of Brain Tumors Using Transfer Learning on Pretrained CNN Models."
- Islam, S., et al. "Brain Tumor Segmentation in MRI Using a Cascaded U-Net Architecture."

For a complete list of references, see the project report.

