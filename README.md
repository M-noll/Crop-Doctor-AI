# 🌿 Plant Doctor: AI-Powered Crop Disease Detection

### **1. Project Overview**
This project is an Artificial Intelligence application developed for identifying crop diseases through image analysis. By leveraging **Deep Learning**, specifically a Convolutional Neural Network (CNN), the system can "see" symptoms on leaves and provide a diagnosis in real-time.

**Developed by:** Tech Maniacs
**Course:** Artificial intelligence
**Problem Solved:** Food security and early disease detection for farmers.

---

### **2. The Solution**
We built a **Computer Vision** model trained on the PlantVillage dataset. The model analyzes patterns, colors, and textures of infected leaves to distinguish between healthy plants and various diseases (e.g., Early Blight, Late Blight, Leaf Mold).



---

### **3. Tech Stack**
* **Language:** Python 3.10+
* **Deep Learning:** TensorFlow / Keras
* **Web Framework:** Streamlit (For the User Interface)
* **Data Handling:** NumPy, Pillow
* **Development Environment:** VS Code

---

### **4. Folder Structure**
```text
CropDoctor/
├── dataset/              # Folder containing images (Tomato, Potato, etc.)
├── venv/                 # Python Virtual Environment
├── train.py              # Script to build and train the AI model
├── app.py                # The Streamlit web application
├── requirements.txt      # List of necessary Python libraries
├── README.md             # Project documentation
└── plant_doctor_model.h5 # The trained model (Generated after training)