# AI & Deep Learning Projects

This repository contains selected **machine learning (ML)** and **deep learning (DL)** projects focused on **image classification**, **model benchmarking**, and **efficient CNN deployment**.  
The projects emphasise **systematic comparison between traditional ML pipelines and modern deep learning models**, with reproducible experiments and clear performance analysis.


## Tech Stack
- Python, PyTorch, scikit-learn
- OpenCV, NumPy, Matplotlib
- CNN fine-tuning and transfer learning




## Projects Overview

### 1. ML vs DL: Aerial Image Classification  
**Traditional Machine Learning vs Deep Learning on Remote Sensing Images**

**Objective**  
Compare handcrafted-feature-based ML approaches with CNN-based deep learning models for multi-class aerial image classification.

**Dataset:**
- 12,000 images, **15 balanced classes**
- Resolution: 256 × 256  
- Categories include Airport, Forest, City, Residential, etc.

**Methods:**
- **Traditional ML**
  - Feature extraction: LBP, SIFT
  - Classifiers: KNN, SVM, Random Forest, XGBoost
- **Deep Learning:**
  - ResNet-18
  - EfficientNet-B0 (ImageNet pretrained)
- 5-fold cross-validation
- Data augmentation and transfer learning

**Key Findings**
- Deep learning models outperform traditional ML in accuracy and robustness
- EfficientNet-B0 provides the **best accuracy–efficiency trade-off**
- Traditional ML remains viable under limited computational resources

📄 Detailed experiments and benchmarks:  
[`ML_vs_DL_Comparison/README.md`](ML_vs_DL_Comparison/README.md)





### 2. Fashion Image Classification (Deep Learning)

**Task:**  
Multi-class fashion image classification into **Accessories, Bags, Clothing, and Shoes** using pretrained CNN architectures.

**Dataset:**
- 8,000 images (2,000 per class)
- Balanced data augmentation

**Models:**
- MobileNetV2  
- ResNet18  
- ResNet50  
- EfficientNet-B0  

**Training & Evaluation:**
- 5-fold cross-validation
- Adam optimiser with early stopping
- Metrics: Accuracy, F1-score, mAP
- Robustness evaluation under Gaussian noise

**Key Findings**
- Lightweight models achieve comparable accuracy to deeper networks
- EfficientNet-B0 shows the strongest robustness with lower computational cost

📄 Full methodology and results:  
[`DL_Fashion_Classification/README.md`](DL_Fashion_Classification/README.md)





### 3.Model Performance Comparison
| Model / Method        | Feature Type        | Accuracy (%) | F1 Score | Params | Notes |
|-----------------------|---------------------|--------------|----------|--------|-------|
| LBP + SVM             | Handcrafted         | 72.4         | 0.71     | -      | Fast, low compute |
| SIFT + Random Forest  | Handcrafted         | 75.8         | 0.74     | -      | Better texture capture |
| XGBoost (SIFT)        | Handcrafted         | 77.1         | 0.76     | -      | Strong classical baseline |
| ResNet-18             | CNN (Pretrained)    | 89.7         | 0.89     | 11.7M  | Stable performance |
| EfficientNet-B0       | CNN (Pretrained)    | 91.3         | 0.91     | 5.3M   | Best accuracy–efficiency trade-off |




---

## Notes
These projects focus on **model comparison, experimental design, and practical trade-offs**, rather than single-model optimisation.
