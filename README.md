# ✍️ Handwritten Character Recognition using Machine Learning

## 📌 Introduction
This project is a final-year coursework project titled **"Xây dựng chương trình nhận diện ký tự viết tay bằng học máy"** (Handwritten Character Recognition using Machine Learning).  
It focuses on applying **Support Vector Machine (SVM)** and **Convolutional Neural Networks (CNN)** to build a system capable of recognizing handwritten digits and characters (both uppercase and lowercase).

---

## 🎯 Objectives
- Build and train machine learning models (SVM, CNN, hybrid approaches) for handwritten character recognition.  
- Collect and preprocess datasets (MNIST + custom Handwritten dataset).  
- Improve recognition accuracy by combining **deep learning features (CNN)** with **traditional features (HOG, Zoning, PCA)**.  
- Develop a **user-friendly application** with frontend (React Native) and backend (Flask + TensorFlow Lite).  
- Evaluate models using metrics such as **Accuracy, Precision, Recall, F1-score, and Inference Time**.  

---

## 📊 Dataset
- **MNIST Dataset**: Subset of 200 samples per class (digits 0–9).  
- **Handwritten Character Dataset**: Created using *Handwritten Character Generator*, with **62 classes**:  
  - 10 digits (0–9)  
  - 26 uppercase letters (A–Z)  
  - 26 lowercase letters (a–z)  
- Total: **~105,000 samples** after preprocessing.

---

## ⚙️ Data Preprocessing
- Label normalization (unify MNIST and custom dataset).  
- Cleaning invalid images (wrong size, empty images).  
- Normalization of pixel values to [0,1].  
- Outlier detection via **Z-score**.  
- One-hot encoding for CNN.  
- Train/test split: **80/20**.  

---

## 🧠 Models Implemented
1. **CNN**  
   - Two convolutional blocks with Conv2D, BatchNorm, MaxPooling, Dropout.  
   - Softmax output layer for 62-class classification.  
   - Optimized with Adam + early stopping.  

2. **CNN + HOG + Zoning**  
   - Multi-input model combining deep learned and handcrafted features.  

3. **SVM**  
   - Trained on flattened image vectors (28x28 → 784 features).  
   - Kernel: RBF.  

4. **SVM + CNN + HOG + PCA**  
   - Combined deep features (CNN) + HOG, reduced with PCA before SVM classification.  

---

## 📈 Results
| Model                  | Accuracy | Precision | Recall | F1-Score | Inference Time |
|------------------------|----------|-----------|--------|----------|----------------|
| **CNN**                | 99.35%   | 99.35%    | 99.34% | 99.34%   | 0.0007s        |
| **CNN + HOG + Zoning** | 92.28%   | 95.13%    | 91.64% | 92.68%   | 0.0002s        |
| **SVM**                | 87.74%   | 97.66%    | 86.59% | 91.07%   | 0.0819s        |
| **SVM + CNN + HOG + PCA** | **99.48%** | **99.51%** | **99.51%** | **99.51%** | 0.0286s |

✅ **Best Model**: SVM + CNN + HOG + PCA → Accuracy 99.48%  
✅ **Fastest Model**: CNN (0.0007s inference, suitable for real-time apps)

---

## 💻 Application
- **Frontend**: React Native (Android app)  
  - Capture images, draw characters, select from gallery.  
  - Display recognition results.  
  - History management.  

- **Backend**: Flask + TensorFlow Lite  
  - Preprocess input images.  
  - Serve ML models (CNN, SVM).  
  - REST API for communication with frontend.  

## 📱 Application UI

<p align="center">
  <img src="docs/ui/99b47f4c0db887e6dea91.jpg" width="250"/>
  <img src="docs/ui/251b39e74b13c14d98022.jpg" width="250"/>
  <img src="docs/ui/abd4d72aa5de2f8076cf3.jpg" width="250"/>
</p>


---

## 🚀 Installation & Usage
```bash
# Clone the repository
git clone https://github.com/Qhuy204/Character-Recognition.git
cd Character-Recognition

# Backend setup
cd backend
pip install -r requirements.txt
python app.py

# Frontend setup
cd frontend
npm install
npm start
