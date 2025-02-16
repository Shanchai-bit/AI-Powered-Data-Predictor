import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.header("AI-Powered Data Predictor!")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file")

if uploaded_file:
    st.success("File uploaded successfully")
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:", df.head(10))

    # Select features and target
    y_label = st.selectbox("What do you want to predict:", df.columns)
    x_label = st.selectbox("Select the features:", df.columns)

    X = df[[x_label]]
    y = df[y_label]

    # Convert categorical target to numbers if needed
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # Train Model
    if "model" not in st.session_state:  
        st.session_state["model"] = LogisticRegression()  # Initialize model

    if st.button("Train Model"):
        st.session_state["model"].fit(X, y)  # Train and store in session state
        st.session_state["trained"] = True  # Set trained flag
        st.success("Model trained successfully!")

    # Prediction
    temp = st.number_input("Enter the value for Temperature:")

    if st.button("Predict"):
        if st.session_state.get("trained", False):  # Check if trained
            prediction = st.session_state["model"].predict([[temp]])[0]
            st.write("Prediction:", prediction)
            #st.write("'1' represent 'Yes'")
            #st.write("'0' represent 'NO'")
        else:
            st.error("Please train the model first before predicting.")

else:
    st.warning("Please upload a dataset to continue.")
