import streamlit as st
import pandas as pd
import pickle
import os
from model import load_and_train_model, predict_species
from visualize import plot_confusion_matrix, plot_feature_importance, plot_pair_plot
from sklearn.model_selection import train_test_split
import logging

# Set up logging for deployment debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure static directory exists
if not os.path.exists("static"):
    os.makedirs("static")

try:
    # Load model and data
    model, X, y, target_names = load_and_train_model()
    logger.info("Model and data loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    st.error("Failed to load model. Please check logs.")

# Generate visualizations (run once or on demand)
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    plot_confusion_matrix(y_test, model.predict(X_test), target_names)
    plot_feature_importance(X, y, X.columns)
    plot_pair_plot(X, y, X.columns, target_names)
    logger.info("Visualizations generated successfully")
except Exception as e:
    logger.error(f"Error generating visualizations: {e}")
    st.error("Failed to generate visualizations. Please check logs.")

# Streamlit app
st.title("Iris Flower Species Classifier")
st.write("Enter the measurements of the iris flower to predict its species.")

# Input fields
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Predict
if st.button("Predict"):
    try:
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                                  columns=X.columns)
        prediction = predict_species(model, input_data)
        species = target_names[prediction]
        st.write(f"Predicted Species: **{species}**")
        logger.info(f"Prediction made: {species}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error("Prediction failed. Please check inputs.")

# Display visualizations
st.header("Model and Data Insights")
for img in ["confusion_matrix.png", "feature_importance.png", "pair_plot.png"]:
    img_path = os.path.join("static", img)
    if os.path.exists(img_path):
        st.image(img_path, caption=img.replace(".png", "").replace("_", " ").title())
    else:
        st.warning(f"Image {img} not found.")