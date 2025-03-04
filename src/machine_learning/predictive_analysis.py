import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from PIL import Image
from src.data_management import load_pkl_file


def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Visualizes the prediction probability results using a bar chart.
    """

    # Create a DataFrame for probabilities
    prob_per_class = pd.DataFrame(
        data={"Probability": [0, 0]}, index=["Healthy", "Infected"]
    )

    # Assign the predicted probability
    prob_per_class.loc[pred_class] = pred_proba
    prob_per_class.loc[prob_per_class.index != pred_class, "Probability"] = (
        1 - pred_proba
    )

    # Round for better display
    prob_per_class = prob_per_class.round(3)
    prob_per_class["Condition"] = prob_per_class.index

    # Generate bar chart
    fig = px.bar(
        prob_per_class,
        x="Condition",
        y="Probability",
        range_y=[0, 1],
        width=600,
        height=300,
        template="seaborn",
        title="Prediction Confidence Levels",
    )
    st.plotly_chart(fig)


def resize_input_image(img, version):
    """
    Resizes an input image to match the average dataset image size for consistency.
    """

    # Load precomputed average image shape
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")

    # Resize image while preserving aspect ratio
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)

    # Normalize pixel values and expand dimensions
    my_image = np.expand_dims(img_resized, axis=0) / 255.0

    return my_image


def load_model_and_predict(my_image, version):
    """
    Loads the trained ML model and performs a prediction on the given image.
    """

    # Load the pre-trained model
    model = load_model(f"outputs/{version}/mildew_detector_mobilenetv2.h5")

    # Perform prediction
    pred_proba = model.predict(my_image)[0, 0]

    # Class mapping
    target_map = {0: "Healthy", 1: "Infected"}
    pred_class = target_map[int(pred_proba >= 0.5)]

    # Adjust probability to always correspond to predicted class
    if pred_class == "Infected":
        pred_proba = 1 - pred_proba

    # Display the result
    st.write(
        f"### **Prediction Result:**\n"
        f"The AI analysis indicates that the leaf is **{pred_class.lower()}** "
        f"with a confidence level of **{pred_proba:.3f}**."
    )

    return pred_proba, pred_class
