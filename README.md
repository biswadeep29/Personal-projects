# 🔮 Streamlit ML Showcase  
### Diamond Pricing & Land Terrain Classification

Welcome to my interactive **Streamlit** portfolio app — a minimalist yet powerful demonstration of two machine learning models in action. Real-time predictions, clean UI, and smart workflows, all rolled into one.

---

## 🏠 Home Page

A simple introduction about me — what I do, what I build, and why I love working with data. Think of it as the lobby to my machine learning museum.

---

## 💎 Project 1: Diamond Price Prediction

Ever wondered what a diamond is worth before walking into a store? This app estimates a diamond’s price based on key physical and quality attributes:

- **Carat**
- **Cut**
- **Color**
- **Clarity**
- **Depth**
- **Table**
- **Length**, **Width**, and **Height**

### 🧠 Under the Hood:

- **Model:** A trained regression model (e.g., Linear Regression, XGBoost, etc.)
- **Encoding:**
  - `cut` is label-encoded.
  - `color` and `clarity` are one-hot encoded using `OneHotEncoder`.
- **UI:** Inputs collected via sliders and dropdowns using `st.form`.
- **Output:** A precise price prediction shown instantly upon submission.

### 🧪 Try it out:

Input values like:
- `Carat: 0.79`
- `Cut: Ideal`
- `Color: G`
- `Clarity: SI1`  
...and get an instant estimate of your stone's market value.

---

## 🌍 Project 2: Land Terrain Classification

A vision-based classifier powered by **Convolutional Neural Networks (CNNs)** that predicts the type of landscape in an uploaded image. It supports classification into the following categories:

- 🏙️ Buildings  
- 🌳 Forest  
- ❄️ Glacier  
- 🏔️ Mountain  
- 🌊 Sea  
- 🛣️ Street  

### 🧠 Model Workflow:

- Two CNN models power the predictions:
  - One trained with **data augmentation**
  - Another **fine-tuned** for accuracy
- Both outputs are compared for final prediction
- A warning is displayed if confidence is low
- Otherwise, you get a clean prediction with probability

### 🧪 Try it out:

Upload a photo of a snowy region — the model may identify it as “Glacier” with over **93% confidence**. Magic? Nah. Just machine learning.
