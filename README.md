🔮 Streamlit ML Showcase: Diamond Pricing & Land Terrain Classification
Welcome to my mini ML portfolio app built using Streamlit! This web app demonstrates two machine learning projects with interactive interfaces and real-time predictions.

🏠 Home Page
Just a brief introduction about me — who I am, what I do, and why I love data. You'll get a feel for my work and this project at a glance.

💎 Project 1: Diamond Price Prediction
A regression-based web app that estimates the price of a diamond based on various features such as:

Carat

Cut

Color

Clarity

Depth

Table

Dimensions (length, width, height)

🧠 Behind the Scenes:
Model: Trained regression model (likely Linear Regression, XGBoost, etc.)

Encoding:

cut encoded using label mapping

color and clarity encoded using OneHotEncoder

Input: Collected via st.form and sliders/dropdowns

Output: Predicted price shown instantly after submission

🧪 Example:
Enter values like a 0.79-carat diamond with Ideal cut, G color, SI1 clarity, and see the predicted price in real-time.

🌍 Project 2: Land Terrain Classification - CNN Model
A classification model that takes an uploaded image and identifies the terrain type. It supports 6 classes:

Buildings

Forest

Glacier

Mountain

Sea

Street

🧠 Model Workflow:
Two CNN models are used:

One with data augmentation

One fine-tuned for improved accuracy

Predictions from both models are compared

If confidence is low, a warning is shown

Otherwise, the predicted class and probability are displayed

🧪 Example:
Upload a picture of a snowy area, and the model may predict “Glacier” with 93.12% confidence.
