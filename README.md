# 🌼 Flower Classifier Web App

This is a **Streamlit** web application that classifies flower images into one of five categories using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**.

## 🚀 Live Demo

🔗 [Flower Classifier](https://flower-classifier1.streamlit.app/)

---

## 📂 Features

- 🌼 Classifies flower images into:
  - Daisy
  - Dandelion
  - Rose
  - Sunflower
  - Tulip

- 📷 Upload an image and receive:
  - Top 3 predicted flower types
  - Confidence scores with progress bars
  - Final predicted label with confidence badge

- 🧠 Learn about the model and dataset from the **About** section

---

## 📊 Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow / Keras
- **Input size**: 180 × 180 RGB
- **Dataset**: [TensorFlow Flowers Dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers)
- **Classes**: 5 (Daisy, Dandelion, Rose, Sunflower, Tulip)

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/flower-classifier-app.git
cd flower-classifier-app
````

### 2. Install Dependencies

Use pip or a virtual environment:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit tensorflow pillow matplotlib numpy
```

### 3. Add Your Model

Make sure your trained Keras model is saved as `flower_classifier_model.keras` and placed in the project root directory.

### 4. Run the App Locally

```bash
streamlit run app.py
```

---

## 📁 File Structure

```
flower-classifier-app/
├── app.py                     # Main Streamlit app
├── flower_classifier_model.keras  # Your trained model (not in repo)
├── requirements.txt           # Required Python packages
└── README.md                  # Project documentation
```

---

## ✅ Example Usage

1. Go to the **Predict** page.
2. Upload an image of a flower (JPG/PNG).
3. See the top 3 predictions with confidence scores.
4. Learn more on the **About Model** page.

---

## 📸 Screenshots

### 🏠 Home Page

![Home Screenshot](assets/home.png)

### 📷 Prediction Page

![Predict Screenshot](assets/predict.png)

---

## 🧑‍💻 Author

**Thogata Madam Hari Ram**
📧 [tmhariram@gmail.com](mailto:tmhariram@gmail.com)
🔗 [Portfolio Website](https://hariram130303.github.io/Portfolio/)
🔗 [LinkedIn](https://linkedin.com/in/hari-ram-thogata-madam)
🔗 [GitHub](https://github.com/hariram130303)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 💡 Tips

* For best results, upload clear images with the flower centered.
* Supported formats: JPG, JPEG, PNG
* This app runs fully in your browser; no data is sent to a server.
