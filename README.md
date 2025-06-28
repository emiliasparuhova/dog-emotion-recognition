# Dog Emotion Recognition

**Note:** The Jupyter notebook is too large to display directly on GitHub. Please view it using nbviewer: [https://nbviewer.org/github/emiliasparuhova/dog-emotion-recognition/blob/main/dog_emotions.ipynb](https://nbviewer.org/github/emiliasparuhova/dog-emotion-recognition/blob/main/dog_emotions.ipynb)

## Project Overview

A sophisticated machine learning model designed to analyse and classify dog emotions into four categories: **happy**, **angry**, **sad**, and **relaxed**, achieving an impressive accuracy rate of approximately **78%**.

The project involved rigorous data cleaning using YOLOv3 to detect and extract clear dog faces, filtering out images with people or unclear features, and removing duplicates with MD5 hashing. To balance the dataset, data augmentation techniques such as rotation, shear, and flips were applied, carefully adjusting parameters to avoid distortion.

Model training progressed from a baseline CNN to advanced transfer learning with VGG16 and finally MobileNet, fine-tuning the last 20 layers while freezing earlier ones. Using early stopping and stratified splits, the final model achieved 78% accuracy in recognising dog emotions.

## Motivation

The primary motivation behind this project is to investigate the feasibility of predicting dog emotions from pictures using machine learning. Understanding dog emotions is challenging with current technology, and this project aims to determine whether advanced machine learning algorithms can create a reliable tool for this purpose.

## Dataset

The project uses the [Dog Emotion dataset](https://www.kaggle.com/datasets/danielshanbalico/dog-emotion) from Kaggle, which contains:
- **4,000 initial images** of dogs categorized into four emotional states
- **Balanced distribution** with 1,000 samples per emotion category
- **High-quality images** manually annotated by the dataset creator

### Data Quality Standards

The dataset underwent rigorous cleaning to ensure:
- **Presence of a Dog**: Images must clearly feature a dog
- **Visibility of Dog Faces**: The dog's face should be sufficiently visible and identifiable
- **Exclusion of People**: Images with people present were removed to prevent bias
- **Duplicate Removal**: MD5 hashing was used to identify and remove duplicate images

## Methodology

### Data Preparation

1. **Object Detection with YOLOv3**
   - Used pre-trained YOLOv3 model for dog detection
   - Filtered out images without clear dog presence
   - Extracted dog faces with high confidence threshold (0.9)

2. **Data Cleaning**
   - Removed 718 images that didn't meet quality standards
   - Final dataset: 3,278 high-quality images
   - Maintained balanced distribution across emotion categories

3. **Data Augmentation**
   - Applied rotation, shear, width/height shifts, and horizontal flips
   - Used oversampling to balance the refined dataset
   - Careful parameter tuning to avoid distortion of facial features

### Model Development

The project explored multiple model architectures with progressive improvements:

#### 1. Baseline CNN
- Simple convolutional neural network
- **Accuracy: 34%**
- Showed signs of overfitting

#### 2. CNN with Data Augmentation
- Added data augmentation to reduce overfitting
- **Accuracy: 36%**
- Improved generalization but still limited performance

#### 3. CNN with VGG16 Transfer Learning
- Used pre-trained VGG16 convolutional base
- Frozen base layers with custom classifier
- **Accuracy: 62%**
- Significant improvement in feature extraction

#### 4. MobileNet (Frozen Base)
- Leveraged MobileNet architecture
- Applied to refined dataset with extracted dog faces
- **Accuracy: 70%**
- Better efficiency and performance

#### 5. MobileNet with Fine-Tuning (Final Model)
- Unfroze last 20 layers of MobileNet base
- Applied early stopping and dropout regularization
- **Final Accuracy: 78%**
- Best overall performance with balanced precision and recall

### Technical Implementation

- **Framework**: TensorFlow/Keras
- **Object Detection**: YOLOv3 with COCO weights
- **Image Processing**: OpenCV, PIL
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Optimization**: Early stopping, stratified sampling

## Results

### Final Model Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Angry   | 0.83      | 0.81   | 0.82     | 135     |
| Happy   | 0.82      | 0.73   | 0.77     | 135     |
| Relaxed | 0.63      | 0.85   | 0.72     | 134     |
| Sad     | 0.95      | 0.75   | 0.84     | 134     |

**Overall Accuracy: 78%**
**Macro Average F1-Score: 0.79**

### Key Achievements

- Successfully developed a reliable dog emotion classification system
- Achieved balanced performance across all emotion categories
- Implemented robust data cleaning and preprocessing pipeline
- Demonstrated effective use of transfer learning and fine-tuning techniques

## Repository Contents

- `dog_emotions.ipynb` - Complete Jupyter notebook with detailed analysis and model development
- `dog_emotions.html` - HTML version of the notebook for easy viewing

## Installation and Usage

### Prerequisites

```bash
pip install tensorflow opencv-python pillow numpy pandas matplotlib seaborn scikit-learn
```

### Additional Requirements

- YOLOv3 weights and configuration files
- COCO class names file
- Sufficient computational resources for model training

### Running the Project

1. Clone the repository
2. Download the required YOLOv3 files
3. Install dependencies
4. Run the Jupyter notebook or view the HTML version

## References

1. Dog emotion. (2023, February 9). Kaggle. https://www.kaggle.com/datasets/danielshanbalico/dog-emotion
2. Colino, S. (2021, October 1). Yes, dogs can "catch" their owners' emotions. National Geographic.
