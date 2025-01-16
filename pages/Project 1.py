import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import os

model_path = "pages/diamond price prediction model 2.pkl"
model = joblib.load(model_path)

tab1, tab2 = st.tabs(["Project", "Theory"])

with tab1:
    st.title("Diamond Price Prediction App")
    st.write("Enter the following information:")

    # Define OneHotEncoders for color and clarity
    color_encoder = OneHotEncoder(sparse_output=False, categories=[['D', 'E', 'F', 'G', 'H', 'I', 'J']])
    clarity_encoder = OneHotEncoder(sparse_output=False, categories=[['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']])

    # Fit encoders (assuming these categories match training data)
    color_encoder.fit(np.array(['D', 'E', 'F', 'G', 'H', 'I', 'J']).reshape(-1, 1))
    clarity_encoder.fit(np.array(['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']).reshape(-1, 1))

    form_values = {
        "carat": None,
        "cut": None,
        "color": None,
        "clarity": None,
        "depth": None,
        "table": None,
        "length": None,
        "width": None,
        "height": None,
    }

    with st.form(key="diamond information"):
        # User inputs
        form_values['carat'] = st.slider("Enter weight of the diamond in carats (mean = 0.79): ", 0.2, 1.5, 0.79, 0.01)
        form_values['cut'] = st.selectbox("Enter quality of the diamond's cut (max occ = Ideal): ", ['Ideal', 'Premium', 'Good', 'Very Good', 'Fair'], 0)
        form_values['color'] = st.selectbox("Enter Color grade of the diamond (max occ = G): ", ['E', 'I', 'J', 'H', 'F', 'G', 'D'], 5)
        form_values["clarity"] = st.selectbox('Enter clarity of the diamond (max occ = SI1): ', ['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF'], 1)
        form_values['depth'] =  61.7
        form_values['table'] =  57.4
        form_values['length'] = 5.73
        form_values['width'] = 5.7
        form_values['height'] = 3.5

        # Submit button
        submit_button = st.form_submit_button()
        if submit_button:
            # Display entered information
            with st.expander("The given info about the diamond are:"):
                for key, value in form_values.items():
                    st.write(f'{key} : {value}')

            # Process the input for the model
            cut_mapping = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
            cut_encoded = cut_mapping[form_values['cut']]

            # OneHotEncode color and clarity
            color_encoded = color_encoder.transform(np.array([form_values['color']]).reshape(-1, 1)).flatten()
            clarity_encoded = clarity_encoder.transform(np.array([form_values['clarity']]).reshape(-1, 1)).flatten()

            # Combine inputs
            processed_input = [
                form_values['carat'],
                cut_encoded,
                *color_encoded,
                *clarity_encoded,
                form_values['depth'],
                form_values['table'],
                form_values['length'],
                form_values['width'],
                form_values['height']
            ]

            # Convert to NumPy array and reshape
            input_array = np.array(processed_input).reshape(1, -1)

            # Predict the price
            predicted_price = model.predict(input_array)[0]

            # Display predicted price
            st.success(f"The predicted price of the diamond is: ${predicted_price:,.2f}")
    
    with tab2:
        st.title("Theory Behind the Diamond Price Prediction Model")

        st.write("""
        ### Overview of the Dataset
        The dataset used for training the diamond price prediction model includes the following features:
        - Carat: Weight of the diamond.
        - Cut: Quality of the diamond's cut (e.g., Ideal, Premium).
        - Color: Grade of the diamond's color (e.g., D, E, F).
        - Clarity: Clarity grade of the diamond (e.g., SI1, VVS1).
        - Depth: Total depth percentage.
        - Table: Width of the diamond's top facet as a percentage of the total width.
        - Length, Width, Height: Dimensions of the diamond in millimeters.

    #### Insights from Feature Analysis
        - Carat: Strongly correlated with price, making it one of the most critical predictors.
        - Cut, Color, and Clarity: Significant contributors to price variations, as they define the diamond's overall quality.
        - Depth, Table, Length, Width, and Height: While part of the dataset, these features were observed to have minimal impact on price. Even when their values changed, the price predictions remained nearly constant.

        #### Model Simplifications
        To enhance efficiency:
        - After training the model, it was found that changes in depth, table, length, width, and height had minimal impact on the price, so their values were treated as constants.
        - The focus was retained on features with significant predictive power, such as carat, cut, color, and clarity.

        #### Key Takeaways
        - **Simplification Benefit**: Reducing less impactful features speeds up predictions and reduces the risk of overfitting.
        - **Critical Features**: Carat, cut, color, and clarity dominate the diamond price determination process.

        ### Dataset Visualizations
        Below are some visualizations to illustrate the dataset features and their relationships with price:
        """)

        file_path = os.path.join(os.getcwd(),"data","diamonds_updated.csv")
        df = pd.read_csv(file_path)
        st.subheader("The given dataset is : ")
        st.dataframe(df.head())

        # Placeholder for graphsst.subheader("2. Distribution of Price by Carat")
        st.write("A graph showing price vs. carat weight.")
        st.image(os.path.join(os.getcwd(),"static","caratvsprice.png"),width=400)  # getcwd = get current working directory
        st.divider()

        st.subheader("3. Price Variation by Cut, Color, and Clarity")
        st.write("Bar plots to depict price distribution across categorical features.")
        st.image(os.path.join(os.getcwd(),"static","color.png"),width=400)
        st.image(os.path.join(os.getcwd(),"static","cut.png"),width=400)
        st.image(os.path.join(os.getcwd(),"static","clarity.png"),width=400)
        st.divider()
