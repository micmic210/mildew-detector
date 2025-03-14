import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities,
)


def page_mildew_detector_body():
    """Allows users to upload cherry leaf images for mildew detection
    using a trained ML model."""
    st.write(f"## Mildew Detection in Cherry Leaves")

    st.info(
        f"Upload images of cherry leaves to determine whether they are "
        f"**healthy** or **infected with powdery mildew**. You can analyze "
        f"multiple images simultaneously and download the results as a CSV "
        f"report."
    )

    st.write(
        f"For testing, you can download sample infected and healthy leaf "
        f"images from [this Kaggle dataset]"
        f"(https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)."
    )

    st.write(f"---")

    # Image upload section
    st.write(
        f"**Upload a clear image of a cherry leaf (or multiple images).**"
    )
    images_buffer = st.file_uploader(
        f" ", type=["jpeg", "jpg", "png"], accept_multiple_files=True
    )

    if images_buffer:
        df_report = pd.DataFrame([])

        for image in images_buffer:
            img_pil = Image.open(image)
            st.info(f"Cherry Leaf Sample: **{image.name}**")

            img_array = np.array(img_pil)
            st.image(
                img_pil,
                caption=(
                    f"Image Size: {img_array.shape[1]}px width x "
                    f"{img_array.shape[0]}px height"
                ),
            )

            # Resize the image for model input
            version = "v1"
            resized_img = resize_input_image(img=img_pil, version=version)

            # Predict using ML model
            pred_proba, pred_class = load_model_and_predict(
                resized_img, version=version
            )
            plot_predictions_probabilities(pred_proba, pred_class)

            # Store results in a dataframe
            df_report = pd.concat(
                [
                    df_report,
                    pd.DataFrame(
                        [{"Image Name": image.name, "Prediction": pred_class}]
                    ),
                ],
                ignore_index=True,
            )

        # Display and allow download of results
        if not df_report.empty:
            st.success(f"Analysis Report")
            st.table(df_report)
            st.markdown(
                download_dataframe_as_csv(df_report),
                unsafe_allow_html=True,
            )

    st.write(
        f"For additional details, visit the "
        f"[Project README](https://github.com/micmic210/mildew-detector/blob/"
        f"main/README.md)."
    )
