import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import base64


# Function to safely load test evaluation from evaluation.pkl
def load_test_evaluation(version):
    file_path = f"outputs/{version}/evaluation.pkl"
    try:
        with open(file_path, "rb") as file:
            test_eval = pickle.load(file)

        if isinstance(test_eval, dict):
            test_eval = list(test_eval.values())
        return test_eval  # Expecting [loss, accuracy]

    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading evaluation.pkl: {e}")
        return None


# Function to Load PCA CSV
def load_pca_data(version):
    """Load PCA CSV data"""
    pca_path = f"outputs/{version}/pca_data.csv"
    try:
        return pd.read_csv(pca_path)
    except FileNotFoundError:
        st.error(f"PCA data not found: {pca_path}")
        return None


# Function to Encode Image as Base64 for HTML Display
def encode_image(image_path):
    """Convert local image to base64 for proper HTML display in Streamlit."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# Function to Display Fixed Size Images in HTML
def display_image_fixed_size(image_path, caption, width=700):
    """Display image with a fixed size using base64 encoding to avoid path issues."""
    base64_img = encode_image(image_path)
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{base64_img}" width="{width}">
            <p style="font-size:14px;">{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Define the main function to display model performance metrics
def page_ml_performance_metrics():
    """Displays the performance metrics of the trained ML model."""

    version = "v1"

    st.write(f"## Model Performance & Evaluation")

    st.info(
        f"This section presents how the dataset was split for training, "
        f"how well the model performed, and key performance metrics."
    )

    # PCA Feature Space Visualization
    st.write(f"### Feature Space Visualization (PCA)")

    pca_data = load_pca_data(version)
    if pca_data is not None:
        fig_pca = px.scatter(
            pca_data,
            x="PC1",
            y="PC2",
            color="Label",
            title=f"PCA: Feature Space of Dataset",
            labels={
                "PC1": "Principal Component 1",
                "PC2": "Principal Component 2",
            },
            width=700,
            height=500,
        )
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                st.plotly_chart(fig_pca)

        # Insights for PCA
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                st.info(
                    f"**PCA Insights:**\n"
                    f"- PCA reduces high-dimensional image data into "
                    f"a **2D representation**.\n"
                    f"- Each dot represents a **sample leaf** from the dataset.\n"
                    f"- **Healthy (dark blue)** and **Infected (light blue)** "
                    f"samples should ideally form separate clusters.\n"
                    f"- If **overlapping occurs**, it suggests that more "
                    f"feature extraction may be needed."
                )

    st.write(f"---")

    # Dataset Split & Class Distribution
    st.write(f"### Dataset Split & Class Distribution")
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.image(
                f"outputs/{version}/labels_distribution.png",
                caption=(
                    f"Class Distribution in Train, Validation, and Test Sets"
                ),
                use_container_width=True,
            )

    # Dataset Insights
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.info(
                f"**Dataset Insights:**\n"
                f"- The dataset is split into **Training (70%)**, "
                f"**Validation (10%)**, and **Test (20%)**.\n"
                f"- Each split contains a **balanced distribution** of "
                f"Healthy and Infected samples.\n"
                f"- This ensures the model learns effectively without bias."
            )

    st.write(f"---")

    # Classification Reports (Train vs. Test)
    st.write(f"### Classification Report (Train vs. Test)")

    train_report_path = (
        f"outputs/{version}/classification_report_train.csv"
    )
    test_report_path = (
        f"outputs/{version}/classification_report_test.csv"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"#### Train Set Classification Report")
        train_report = pd.read_csv(train_report_path)
        st.dataframe(train_report, height=300)

    with col2:
        st.write(f"#### Test Set Classification Report")
        test_report = pd.read_csv(test_report_path)
        st.dataframe(test_report, height=300)

    # Classification Report Insights
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.info(
                f"**Classification Report Insights:**\n"
                f"- Precision, Recall, and F1-Score should be **high and "
                f"consistent** across train and test sets.\n"
                f"- A significant drop in test performance may indicate "
                f"**overfitting**, meaning the model memorized training "
                f"data but struggles with unseen data.\n"
                f"- A balanced model should have a **small difference** "
                f"between train and test scores."
            )

    st.write(f"---")

    # Confusion Matrix (Train vs. Test)
    st.write(f"### Confusion Matrix (Train vs. Test)")

    col1, col2 = st.columns(2)

    with col1:
        display_image_fixed_size(
            f"outputs/{version}/confusion_matrix_train.png",
            f"Confusion Matrix (Train Set)",
            width=350,
        )

    with col2:
        display_image_fixed_size(
            f"outputs/{version}/confusion_matrix_test.png",
            f"Confusion Matrix (Test Set)",
            width=350,
        )

    # Confusion Matrix Insights
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.info(
                f"**Confusion Matrix Insights:**\n"
                f"- **True Positives (TP) & True Negatives (TN):** "
                f"Correctly classified samples.\n"
                f"- **False Positives (FP) & False Negatives (FN):** "
                f"Misclassified samples.\n"
                f"- A high **FN rate** suggests the model struggles "
                f"to detect infections.\n"
                f"- Ideally, TP and TN should be high, while FP and FN "
                f"should be low."
            )

    st.write(f"---")


    # Model Learning Curves (Accuracy & Loss)
    st.write(f"### Model Training Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            f"outputs/{version}/accuracy_curve_softmax.png",
            caption=f"Model Training Accuracy",
            use_container_width=True,
        )

    with col2:
        st.image(
            f"outputs/{version}/loss_curve_softmax.png",
            caption=f"Model Training Loss",
            use_container_width=True,
        )

    # Learning Curves Insights
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.info(
                f"**Learning Curve Insights:**\n"
                f"- The **Loss Curve** should steadily decrease, indicating "
                f"the model is learning correctly.\n"
                f"- The **Accuracy Curve** should increase and stabilize, "
                f"with minimal overfitting.\n"
                f"- Large gaps between **training and validation curves** "
                f"suggest **overfitting** and should be minimized."
            )

    st.write(f"---")

    # Prediction Probability Histogram
    st.write(f"### Prediction Probability Histogram (Test Set)")
    display_image_fixed_size(
        f"outputs/{version}/histogram_test.png",
        f"Prediction Probabilities Histogram (Test Set)",
        width=700,
    )

    # Histogram Insights
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.info(
                f"**Histogram Insights:**\n"
                f"- The histogram represents the model's confidence "
                f"in its predictions.\n"
                f"- Ideally, predictions should be **clearly separated** "
                f"at the 0.5 probability threshold.\n"
                f"- **Overlapping regions** indicate uncertainty in "
                f"classifications, which may suggest further model tuning."
            )

    st.write(f"---")

    # ROC Curve
    st.write(f"### ROC Curve")
    display_image_fixed_size(f"outputs/{version}/roc_curve.png", f"ROC Curve", width=700)

    # ROC Curve Insights
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.info(
                f"**ROC Curve Insights:**\n"
                f"- The **AUC score** (Area Under Curve) measures the "
                f"model’s ability to separate classes.\n"
                f"- A value close to **1.0** indicates excellent performance, "
                f"while a value near **0.5** suggests poor discrimination.\n"
                f"- A well-performing model should have an AUC "
                f"**greater than 0.90**."
            )

    st.write(f"---")

    # Final Model Performance Table
    st.write(f"### Final Model Performance on Test Set")

    test_eval = load_test_evaluation(version)

    if isinstance(test_eval, (list, tuple)) and len(test_eval) == 2:
        df_test_eval = pd.DataFrame(
            {
                "Metric": ["Loss", "Accuracy"],
                "Value": [
                    round(test_eval[0], 4),
                    round(test_eval[1], 4),
                ],
            }
        )

        with st.container():
            col1, col2, col3 = st.columns([1, 5, 1])
            with col2:
                st.table(df_test_eval)

    else:
        st.error(f"Unexpected format in evaluation.pkl. Verify file contents.")

    st.write(
        f"For additional details, visit the "
        f"[Project README](https://github.com/micmic210/mildew-detector/blob/"
        f"main/README.md)."
    )
