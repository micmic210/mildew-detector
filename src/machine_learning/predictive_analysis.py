import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from PIL import Image
from src.data_management import load_pkl_file


def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Plot prediction probability results in a bar chart.
    """

    # Define class labels
    class_labels = ["Healthy", "Infected"]

    # Create a DataFrame for probabilities
    prob_per_class = pd.DataFrame(
        data=[1 - pred_proba, pred_proba],  # Ensure correct probability mapping
        index=class_labels,
        columns=["Probability"],
    )

    prob_per_class = prob_per_class.round(3)
    prob_per_class["Diagnostic"] = prob_per_class.index

    # Generate the bar chart
    fig = px.bar(
        prob_per_class,
        x="Diagnostic",
        y="Probability",
        range_y=[0, 1],
        width=600,
        height=300,
        template="seaborn",
    )

    # Ensure unique keys to avoid duplicate elements
    st.plotly_chart(fig, use_container_width=True, key=f"plot_{pred_class}_{id(fig)}")

def resize_input_image(img, version):
    """
    Reshape image to match the expected input size.
    """

    # Load precomputed image shape
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")

    # Resize while maintaining aspect ratio
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)

    # Normalize and reshape
    my_image = np.expand_dims(img_resized, axis=0) / 255.0

    return my_image


def load_model_and_predict(my_image, version):
    """
    Load trained model and make a prediction.
    """

    # Load the pre-trained model
    model = load_model(f"outputs/{version}/softmax.h5")

    # Get the predicted probability
    pred_proba = model.predict(my_image)[0, 0]

    # Reverse probability mapping
    pred_proba = 1 - pred_proba

    # Assign class based on adjusted probability
    pred_class = "Infected" if pred_proba >= 0.5 else "Healthy"

    # Display results
    st.write("### Prediction Result:")
    st.write(f"The AI analysis indicates the sample leaf is **{pred_class.lower()}**")

    return pred_proba, pred_class
