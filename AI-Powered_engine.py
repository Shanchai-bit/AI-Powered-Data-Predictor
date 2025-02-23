import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

# App Title
st.header("AI-Powered Data Predictor!")

# File uploader for CSV dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        # Load dataset
        df = pd.read_csv(uploaded_file)
        df.index = range(1, len(df)+1)  # Start index from 1

        # Check if the dataset is empty or has no columns
        if df.empty or len(df.columns) == 0:
            st.error("Uploaded file is empty or has no columns.")
        else:
            st.success("File uploaded successfully!")
            st.write("Data Preview:", df.head(10))  # Display first 10 rows

            # Select target variable (the column we want to predict)
            y_label = st.selectbox("What feature do you want to predict?", df.columns)

            # Check for missing values in the target column
            if df[y_label].isnull().any():
                st.error(f"The target column '{y_label}' contains missing values. Please clean your data.")
            else:
                # Feature selection mode
                mode = st.radio("Select Mode:", ['Automatic', 'Manual'])
                st.write(mode, "mode selected.")

                if mode == "Automatic":
                    # Select all columns except target as features
                    x_label = df.columns.drop(y_label).tolist()
                else:
                    # User manually selects features
                    x_label = st.multiselect("Select the features:", df.columns.drop(y_label))
                    st.write(f"Selected features: {x_label}")

                # Ensure at least one feature is selected and target is not in features
                if x_label and y_label not in x_label:
                    X = df[x_label].copy()  # Feature variables
                    y = df[y_label].copy()  # Target variable

                    # Encode categorical target variable if necessary
                    encoder = None
                    if y.dtype == "object":
                        if y.nunique() == 2:
                            encoder = LabelEncoder()
                            y = encoder.fit_transform(y)  # Convert categorical labels to numerical
                            model = LogisticRegression()  # Use Logistic Regression for classification
                        else:
                            model = DecisionTreeClassifier()
                    else:
                        model = LinearRegression()  # Default regression model

                    # Encode categorical features
                    encoders = {}  # Dictionary to store encoders
                    for col in X.select_dtypes(include=["object"]).columns:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col])  # Convert categorical features to numerical
                        encoders[col] = le  # Store encoder for later use

                    # Train Model Button
                    if st.button("Train Model", type="primary"):
                        try:
                            # Train the initial model
                            model.fit(X, y)
                            score = model.score(X, y)  # Get model accuracy

                            # Adjust model type for regression tasks based on performance
                            if isinstance(model, LinearRegression):
                                if score < 0.8:
                                    model = Lasso()  # Use Lasso for low scores
                                elif score > 0.9:
                                    model = Ridge()  # Use Ridge for high scores
                                else:
                                    model = ElasticNet()  # Use ElasticNet for moderate scores
                                model.fit(X, y)  # Retrain with selected model
                                score = model.score(X, y)  # Get model accuracy

                            if isinstance(model, DecisionTreeClassifier):
                                plt, fig = plt.subplots()
                                plot_tree(model, filled=True, feature_names=x_label,ax=fig)
                                st.pyplot(plt)
                                if score <= 0.8:
                                    model = RandomForestClassifier()  # Use Random Forest for classification
                                model.fit(X, y)  # Retrain with selected model  
                                score = model.score(X, y)  # Get model accuracy  

                            # Store trained model and metadata in session state
                            st.session_state["trained_model"] = model
                            st.session_state["feature_names"] = x_label
                            st.session_state["target_encoder"] = encoder
                            st.session_state["feature_encoders"] = encoders

                            st.success("Model trained successfully!")
                            st.write(f"Model Accuracy: {score:.2f}")  # Display accuracy

                        except Exception as e:
                            st.error(f"Error during model training: {str(e)}")  # Handle training errors

                    # User input for predictions
                    st.subheader("Make a Prediction")
                    user_inputs = {}  # Dictionary to store user inputs
                    for feature in x_label:
                        user_inputs[feature] = st.text_input(f"Enter value for {feature}:")  # Input field for each feature

                    # Predict Button
                    if st.button("Predict", type="primary"):
                        try:
                            if "trained_model" not in st.session_state:
                                st.error("Please train the model first before making a prediction.")
                            else:
                                # Convert user inputs to DataFrame
                                input_df = pd.DataFrame([user_inputs])

                                # Encode categorical inputs using stored encoders
                                for col, le in st.session_state["feature_encoders"].items():
                                    if col in input_df.columns:
                                        try:
                                            input_df[col] = le.transform(input_df[col])
                                        except ValueError:
                                            st.error(f"Unseen category in feature '{col}'. Please check your input.")
                                            
                                # Handle missing or invalid values
                                if input_df.isnull().values.any():
                                    st.error("Invalid input values. Please check your inputs.")
                                else:
                                    # Make prediction
                                    prediction = st.session_state["trained_model"].predict(input_df)[0]

                                    # Decode prediction if target variable was categorical
                                    if st.session_state["target_encoder"]:
                                        prediction = st.session_state["target_encoder"].inverse_transform([int(prediction)])[0]

                                    # Display prediction result
                                    st.write(f"{y_label} Prediction:", prediction)

                        except NotFittedError:
                            st.error("Model is not trained. Please train the model first.")
                        except Exception as e:
                            st.error(f"Error making prediction: {str(e)}")  # Handle prediction errors

                else:
                    st.warning("Please select at least one feature and ensure the target column is not selected as a feature.")

    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")  # Handle file processing errors

else:
    st.warning("Please upload a dataset to continue.")