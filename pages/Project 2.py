import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time

tab1, tab2 = st.tabs(["Project", "Theory"])

model_path = "pages/intel_model_data_aug.pkl"
model_data_aug = joblib.load(model_path)

model_path = "pages/intel_model_fine.pkl"
model_fine = joblib.load(model_path)


def predict_img(img_path,model):
    img = image.load_img(img_path,target_size = (150,150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array,axis = 0)
    img_array /= 255.0

    pred = model.predict(img_array)
    class_idx = np.argmax(pred[0])
    class_labels =  {0: 'buildings',1: 'forest',2: 'glacier',3: 'mountain',4: 'sea',5: 'street'}

    return [class_labels[class_idx], pred[0][class_idx]]

def compare(p1,p2):
    p1_cls = p1[0]
    p1_prob = p1[1]

    p2_cls = p2[0]
    p2_prob = p2[1]

    if (p1_cls == p2_cls):
        avg_prob = (p1_prob + p2_prob)/2
        return [p2_cls,avg_prob]
    
    if (p1_cls != p2_cls):
        if p1_prob > p2_prob:
            return [p1_cls,p1_prob]
        return [p2_cls,p2_prob]


with tab1:
    st.subheader("Land Terrain Classification - CNN Model")
    
    # File uploader for image input
    upload_img = st.file_uploader("Choose an image to upload", type=['png', 'jpg', 'jpeg', 'webp'])
    
    if upload_img:
        st.write("Image provided:")
        
        # Progress bar for operation
        progress_text = "Operation in progress. Please wait."
        progress_bar = st.progress(0, text=progress_text)
        for percent_complete in range(1, 101):
            time.sleep(0.01)
            progress_bar.progress(percent_complete, text=progress_text)
        time.sleep(1)
        progress_bar.empty()
        
        # Display the uploaded image
        st.image(upload_img, width=300)
        
        # Predictions using two models
        prediction_data_aug = predict_img(upload_img, model_data_aug)
        prediction_fine_tuned = predict_img(upload_img, model_fine)
        
        # Final prediction comparison
        final_prediction = compare(prediction_data_aug, prediction_fine_tuned)
        
        # Class labels
        class_labels = {
            0: 'buildings',
            1: 'forest',
            2: 'glacier',
            3: 'mountain',
            4: 'sea',
            5: 'street'
        }
        
        # Handle unclear or low-confidence predictions
        if final_prediction[1] < 0.75:
            st.warning(
                f"The image might not be clear or may not belong to the categories: "
                f"{', '.join(class_labels.values())}"
            )
            st.write("Still, our prediction is:")
        
        # Display final prediction
        st.success(
            f"Predicted class: {class_labels[final_prediction[0]]} "
            f"with probability: {final_prediction[1] * 100:.2f}%"
        )



def display_theory_page():
    """Function to display the theory page for the project in Tab 2."""
    st.title("Project Overview: Land Terrain Classification using Deep Learning")

    st.write("""
    ## Introduction
    This project focuses on building a deep learning model capable of differentiating between various land terrains. The terrains categorized in this model are:
    - Buildings
    - Forest
    - Glacier
    - Mountain
    - Sea
    - Street

    The model leverages convolutional neural networks (CNNs) to extract features from images and classify them into the above categories.

    ## Key Highlights of the Model
    - **Convolutional Layers**: Extract spatial features from the input images.
    - **Batch Normalization**: Improves training stability and speeds up convergence.
    - **MaxPooling Layers**: Reduces the spatial dimensions and prevents overfitting by down-sampling.
    - **Dropout Regularization**: Adds robustness to the model by randomly dropping neurons during training.
    - **Fully Connected Layers**: Perform the final classification into one of the six terrain categories.
    - **Softmax Activation**: Provides probabilities for each terrain category.

    ## Model Performance
    While the model effectively classifies terrains based on the given dataset, there are certain limitations:
    - **Small Training Dataset**: The dataset size is limited, which impacts the model's generalization ability on unseen data.
    - **Potential Overfitting**: Despite regularization techniques like dropout and L2 regularization, the model could exhibit overfitting on the training data.
    
    ## Areas for Improvement
    To enhance the model's performance, the following steps could be considered:
    - Expanding the training dataset with diverse images for each terrain category.
    - Implementing data augmentation techniques to simulate variations in the dataset.
    - Experimenting with more complex architectures like transfer learning using pre-trained models (e.g., ResNet, VGG).

    ## Visual Enhancements
    Adding diagrams can improve the understanding and presentation of the project. Suggested diagrams:
    1. **Data Flow Diagram**: Place this at the beginning to showcase the end-to-end workflow of the project (e.g., data collection, preprocessing, model training, evaluation).
    2. **Terrain Sample Images**: Display sample images of each terrain category to provide visual context.
    3. **Model Training History Plot**: Include graphs showing training and validation accuracy/loss trends over epochs.
    4. **Confusion Matrix**: Add a confusion matrix to visualize the model's performance across all terrain categories.

    ## Conclusion
    Despite its limitations, the model demonstrates the capability of deep learning in classifying diverse land terrains. Future work should focus on expanding the dataset and exploring advanced architectures to further enhance performance.
    """)

with tab2:
    display_theory_page()
