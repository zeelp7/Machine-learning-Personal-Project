# 📦 Image Recognition - Personal Project

Welcome to my Image  recognisition project. This project uses convolutional neural networks (CNNs) to classify images into predefined categories based on pixel data and trained features. This project was a personal exploration into image classification and machine learning. It was exciting to collect the images and experiment with different architectures to see how well the model could learn to differentiate between images of myself and others. I hope to continue building on this experience in future personal projects and apply it to more complex real-world datasets.

---

## 📊 Project Overview

The goal of this project is to build an image classification model that can predict the class of input images. The dataset consists of three primary folders containing labeled images, which represent:

- **"Me Only"** – Images of myself alone.
- **"Me with Others"** – Images of me with other people.
- **"Others"** – Images of other people, without me.

Each image was preprocessed and resized to fit the input requirements of the model. I used various data augmentation techniques to increase the diversity of the dataset and improve the model's performance.

---

## 🛠️ Tech Stack & Tools

- **Python** – Main programming language used.
- **TensorFlow / Keras** – For building and training the deep learning models.
- **Pandas, NumPy** – For handling and manipulating data.
- **Matplotlib** – For visualizations of model training progress.
- **OpenCV** – For image preprocessing and augmentation.

---

## 🧪 ML Models Implemented

1. **Convolutional Neural Network (CNN)**
   - Designed using several convolutional layers, pooling layers, and dense layers to classify images based on pixel data.
2. **Transfer Learning (Pre-trained Models)**
   - Fine-tuned popular models like ResNet and VGG16 for the image classification task, leveraging pre-trained weights on large datasets.
3. **Data Augmentation**
   - Applied random transformations like rotation, zoom, and flipping to improve model generalization by creating variations of the training data.

---

## 📈 Results Summary

- **Best Performing Model:** CNN with transfer learning (ResNet)
- **Highest Test Accuracy:** ~92%
- The "Me with Others" category had the highest variation in results, which improved with additional data augmentation.
- Image preprocessing steps, such as resizing and normalization, played a crucial role in boosting model performance.

---

## 📁 Files Included

- `ImageRecognition.ipynb` – The main Jupyter notebook with steps for data preprocessing, EDA, model building, tuning, and evaluation.
- `README.md` – This file you're reading.
- **Image Folders:**
  - `MeOnly/` – Contains images of me alone.
  - `MeWithOthers/` – Contains images of me with others.
  - `Others/` – Contains images of other people.

---

## 📌 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/image-recognition.git
cd image-recognition

# Open the notebook
jupyter notebook ImageRecognition.ipynb
