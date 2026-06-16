# Automated Detection of Diabetic Retinopathy in Retinal Images Using Deep Learning

## 📌 Project Overview
Diabetic Retinopathy (DR) is a severe medical complication caused by diabetes, leading to permanent vision loss if undetected. Regular retinal screenings are crucial, but manual diagnosis is time-consuming and prone to human error. 

This project introduces an end-to-end Automated Detection and Classification System for Diabetic Retinopathy using state-of-the-art Deep Learning architectures. By leveraging digital fundus photographs, the system categorizes retinal images into distinct severity levels, enabling early, scalable, and efficient diagnostic support.

---

## 🚀 Key Features
* **Automated Multi-Class & Binary Classification:** Dynamic categorization of retinal fundus images based on DR severity.
* **State-of-the-Art Architectures:** Implemented and compared top-performing Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), including optimized models like **MobileNetV3** for edge deployments, **EfficientNet** for high accuracy, and **Vision Transformers (ViTs)** for capturing global context.
* **Advanced Preprocessing:** Enhanced retinal image quality using contrast normalization (CLAHE) and noise reduction to isolate microaneurysms, hemorrhages, and exudates.
* **Explainable AI (XAI):** Integrated feature visualization maps to pinpoint specific lesion locations, ensuring clinical interpretability and trust.

---

## 📂 Repository Structure
* `App-Binaryclass/`: Contains the source code, web application scripts (`app.py`), and trained PyTorch weights (`model_epoch_8.pth`) for the Binary Classification model (Normal vs. Affected).
* `App-Multiclass/`: Contains the source code and web application scripts for the Multi-Class Severity Classification model.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Frameworks:** PyTorch / TensorFlow & Keras
* **Libraries:** OpenCV, NumPy, Pandas, Scikit-learn, Matplotlib
* **Environment:** Jupyter Notebook / Google Colab
