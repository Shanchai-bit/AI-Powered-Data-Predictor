import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

st.header("AI-Powered Data Predictor!")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    st.success("File uploaded successfully")
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head(10))

    # Select target variable
    y_label = st.selectbox("What do you want to predict?", df.columns)

    # Select feature variables
    mode = st.radio("Select Mode:", ['Automatic', 'Manual'])
    if mode == "Automatic":
        st.write("Automatic mode selected.")
        x_label = df.columns[df.columns != y_label].tolist()
    else:
        x_label = st.multiselect("Select the features:", df.columns[df.columns != y_label])
        st.write(f"Selected features: {x_label}")

    if x_label:
        X = df[x_label]
        y = df[y_label]

        # Encode categorical target if necessary
        encoder = None
        if y.dtype == "object":
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)  # Encode categorical target
            model = LogisticRegression()  # Use logistic regression for classification
        else:
            model = LinearRegression()  # Use linear regression for numerical predictions

        # Encode categorical features
        #for col in X.select_dtypes(include=["object"]).columns:
            #le = LabelEncoder()
            #X[col] = le.fit_transform(X[col])

        # Train model
        if st.button("Train Model"):
            model.fit(X, y)
            st.session_state["trained_model"] = model  # Store trained model in session
            st.session_state["feature_names"] = x_label  # Store feature names
            st.session_state["encoder"] = encoder  # Store encoder if classification

            st.success("Model trained successfully!")

            # Show model accuracy
            score = model.score(X, y)
            st.write(f"Model Accuracy: {score:.2f}")

        # User input for prediction
        st.subheader("Make a Prediction")
        inputs = []
        for feature in x_label:
            value = st.text_input(f"Enter value for {feature}:")
            inputs.append(value)

        # Make a prediction
        if st.button("Predict"):
            try:
                if "trained_model" not in st.session_state:
                    st.error("Please train the model first before making a prediction.")
                else:
                    # Convert inputs to DataFrame
                    input_df = pd.DataFrame([inputs], columns=st.session_state["feature_names"])

                    # Convert numeric features
                    for col in input_df.columns:
                        if col in df.select_dtypes(include=["int64", "float64"]).columns:
                            input_df[col] = pd.to_numeric(input_df[col])

                    # Encode categorical features using stored encoder
                    for col in input_df.select_dtypes(include=["object"]).columns:
                        le = LabelEncoder()
                        input_df[col] = le.fit_transform(input_df[col])

                    # Make prediction
                    prediction = st.session_state["trained_model"].predict(input_df)[0]

                    # Decode prediction if target was categorical
                    if st.session_state["encoder"]:
                        prediction = st.session_state["encoder"].inverse_transform([int(prediction)])[0]

                    st.write(f"{y_label} Prediction:", prediction)

            except NotFittedError:
                st.error("Model is not trained. Please train the model first.")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    else:
        st.warning("Please select at least one feature.")

else:
    st.warning("Please upload a dataset to continue.")
