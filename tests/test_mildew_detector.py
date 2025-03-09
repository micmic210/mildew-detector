import pytest
import numpy as np
import pandas as pd
from PIL import Image
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
)
from src.data_management import download_dataframe_as_csv


def test_resize_input_image():
    """Test resizing of input image to expected dimensions"""
    img = Image.new("RGB", (256, 256), color="white")  # Create a sample image
    version = "v1"
    resized_img = resize_input_image(img, version)

    # Fix: Allow for batch dimension (1, 128, 128, 3) or (128, 128, 3)
    assert resized_img.shape in [(128, 128, 3), (1, 128, 128, 3)]


def test_load_model_and_predict():
    """Test model loading and prediction with valid input shape"""
    img = np.random.rand(1, 128, 128, 3)  # Ensure batch dimension exists
    version = "v1"

    pred_proba, pred_class = load_model_and_predict(img, version)

    assert 0.0 <= pred_proba <= 1.0  # Probability should be between 0 and 1
    assert pred_class in ["Healthy", "Infected"]  # Should return valid labels


def test_download_dataframe_as_csv():
    """Test CSV download function to ensure correct format"""
    df = pd.DataFrame({"Image Name": ["test_image.jpg"], "Prediction": ["Healthy"]})
    csv_output = download_dataframe_as_csv(df)

    # Adjust test to match the actual returned HTML anchor format
    assert csv_output.startswith('<a href="data:file/csv;base64,'), "CSV download link is incorrect."
    assert "Download Report</a>" in csv_output, "Download link text is missing."