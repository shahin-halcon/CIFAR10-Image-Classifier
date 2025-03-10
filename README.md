# 📦 Image Classification with CNN Using CIFAR-10

Welcome to my **Image Classification** project! This repository demonstrates how to build and train a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** for classifying images from the popular **CIFAR-10** dataset. The CIFAR-10 dataset is widely used for educational purposes in computer vision and deep learning.

---

## 🚀 Project Overview
In this project, I developed a CNN model to recognize and classify images into one of 10 object categories. The project workflow includes:

- Loading and preprocessing the CIFAR-10 dataset.
- Designing and building a CNN architecture.
- Training and validating the model.
- Evaluating model performance.
- Visualizing results through accuracy/loss plots and confusion matrices.

---

## 📂 Dataset
- **CIFAR-10 Dataset**:
  - 60,000 color images sized 32x32 pixels.
  - 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
  - Split into 50,000 training images and 10,000 testing images.

---

## 🏗️ Model Architecture
The CNN model consists of:
- Convolutional layers with ReLU activation
- MaxPooling layers to reduce dimensionality
- Dropout layers to prevent overfitting
- Dense (fully connected) layers for classification
- A Softmax output layer for multi-class classification

---

## 🔧 Technologies Used
- **Python 3.x**
- **TensorFlow 2.12.0 / Keras**
- **NumPy 1.23.5**
- **Matplotlib 3.7.1**
- **Scikit-learn 1.2.2**

---

## 📋 Requirements and Versions

Here's a list of libraries and versions used in this project:

```txt
tensorflow==2.12.0
numpy==1.23.5
matplotlib==3.7.1
scikit-learn==1.2.2
```

### Optional (if needed):
```txt
jupyterlab==3.6.3
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📊 Results
- Achieved high classification accuracy on the test dataset.
- Plotted training/validation accuracy and loss to visualize learning.
- Displayed confusion matrix for a better understanding of classification performance.

---

## 💡 Key Learnings
- Built a CNN model from scratch for image classification.
- Learned the importance of preprocessing and normalization.
- Applied regularization techniques (like Dropout) to prevent overfitting.
- Visualized performance metrics to evaluate model effectiveness.

---

## 🔮 Future Improvements
- Add data augmentation to increase dataset diversity and boost model generalization.
- Experiment with deeper architectures like ResNet or VGG.
- Perform hyperparameter tuning (optimizer, learning rate, batch size) for improved accuracy.
- Deploy the trained model in a web app for real-time predictions.

---

## 📝 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Image-Classification-CIFAR10-CNN.git
   cd Image-Classification-CIFAR10-CNN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Image_Classification_CNN_Using_CIFAR10.ipynb
   ```

