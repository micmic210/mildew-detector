import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import base64


# Function to safely load test evaluation from evaluation.pkl
def load_test_evaluation(version):
    """Load test evaluation results from evaluation.pkl file."""
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
    """Load PCA CSV data."""
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
    """Display image with a fixed size using base64 encoding."""
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

    st.write("## Model Performance & Evaluation")

    st.info(
        "This section presents how the dataset was split for training, "
        "how well the model performed, and key performance metrics."
    )

    # PCA Feature Space Visualization
    st.write("### Feature Space Visualization (PCA)")

    pca_data = load_pca_data(version)
    if pca_data is not None:
        fig_pca = px.scatter(
            pca_data,
            x="PC1",
            y="PC2",
            color="Label",
            title="PCA: Feature Space of Dataset",
            labels={
                "PC1": "Principal Component 1",
                "PC2": "Principal Component 2"
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
                    "**PCA Insights:**\n"
                    "- The PCA visualization shows some separation between "
                    "Healthy and Infected leaves, but the classes are not "
                    "entirely distinct.\n"
                    "- This suggests that while pixel-level differences "
                    "exist, additional features or transformations may "
                    "improve class separability for ML classification.\n"
                )

    st.write("---")

    # Dataset Split & Class Distribution
    st.write("### Dataset Split & Class Distribution")
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.image(
                f"outputs/{version}/labels_distribution.png",
                caption=("Class Distribution in Train, Validation,"
                         "and Test Sets")
            )

    # Dataset Insights
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.info(
                "**Dataset Insights:**\n"
                "- The dataset is split into **Training (70%)**, "
                "**Validation (10%)**, and **Test (20%)**.\n"
                "- Each split contains a **balanced distribution** of "
                "Healthy and Infected samples.\n"
                "- This ensures the model learns effectively without bias."
            )

    st.write("---")

    # Classification Reports (Train vs. Test)
    st.write("### Classification Report (Train vs. Test)")

    train_report_path = f"outputs/{version}/classification_report_train.csv"
    test_report_path = f"outputs/{version}/classification_report_test.csv"

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Train Set Classification Report")
        train_report = pd.read_csv(train_report_path)
        st.dataframe(train_report, height=300)

    with col2:
        st.write("#### Test Set Classification Report")
        test_report = pd.read_csv(test_report_path)
        st.dataframe(test_report, height=300)

    # Classification Report Insights
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.info(
                "**Classification Report Insights:**\n"
                "- The **train set classification report** shows a balanced "
                "precision, recall, and F1-score, indicating that the model "
                "effectively learns from training data.\n"
                "- The **test set classification report** confirms strong "
                "generalization, with nearly identical metrics across both "
                "classes, ensuring reliable performance on unseen data.\n"
            )

    st.write("---")

    # Confusion Matrix (Train vs. Test)
    st.write("### Confusion Matrix (Train vs. Test)")

    col1, col2 = st.columns(2)

    with col1:
        display_image_fixed_size(
            f"outputs/{version}/confusion_matrix_train.png",
            "Confusion Matrix (Train Set)",
            width=350,
        )

    with col2:
        display_image_fixed_size(
            f"outputs/{version}/confusion_matrix_test.png",
            "Confusion Matrix (Test Set)",
            width=350,
        )

    # Confusion Matrix Insights
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.info(
                f"**Confusion Matrix Insights:**\n\n"
                f"**Train Set Confusion Matrix**\n"
                f"- The model demonstrates **high accuracy** but shows "
                f"**some misclassification** between Healthy and Infected "
                f"leaves.\n"
                f"- Misclassified samples are balanced across both classes, "
                f"indicating **room for minor optimization** but no major "
                f"class bias.\n\n"
                f"**Test Set Confusion Matrix**\n"
                f"- The model performs **exceptionally well** on the test, "
                f"set with **only 2 misclassified samples**.\n"
                f"- **No false negatives** for Healthy leaves and **only 2 "
                f"false positives** for Infected leaves suggest **strong "
                f"generalization**.\n"
            )

    st.write(f"---")

    # Model Learning Curves (Accuracy & Loss)
    st.write(f"### Model Training Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            f"outputs/{version}/accuracy_curve_softmax.png",
            caption=f"Model Training Accuracy",
        )

    with col2:
        st.image(
            f"outputs/{version}/loss_curve_softmax.png",
            caption=f"Model Training Loss",
        )

    # Learning Curves Insights
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.info(
                f"**Learning Curve Insights:**\n"
                f"- **Accuracy Curve** → Both training and validation "
                f"accuracy improved steadily, converging above **99%**.\n"
                f"- **Loss Curve** → The model’s loss consistently decreased "
                f"for both training and validation sets, suggesting effective "
                f"learning without severe overfitting.\n"
                f"- **Generalization** → The small gap between training and "
                f"validation curves suggests **good generalization**, with "
                f"the model maintaining high performance on unseen data.\n"
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
                f"- **Well-Separated Predictions** → The model confidently "
                f"classifies most samples with probabilities near **0 or 1**, "
                f"indicating strong decision boundaries.\n"
                f"- **Minimal Ambiguity** → Very few predictions fall near "
                f"the **0.5 threshold**, suggesting high confidence in "
                f"classifications.\n"
                f"- **Healthy vs. Infected Separation** → The distinct peaks"
                f"for **Healthy (green)** and **Infected (blue)** classes "
                f"confirm that the model effectively differentiates between "
                f"them.\n"
            )

    st.write(f"---")

    # ROC Curve
    st.write(f"### ROC Curve")
    display_image_fixed_size(
        f"outputs/{version}/roc_curve.png",
        f"ROC Curve",
        width=700,
    )

    # ROC Curve Insights
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.info(
                f"**ROC Curve Insights:**\n"
                f"- **Train AUC (0.48) vs. Test AUC (1.00)**: The model "
                f"achieves **perfect discrimination on the test set** but  "
                f"struggles on the training set.\n"
                f"- **Ideal Test Performance**: AUC = 1.00 on the test set "
                f"indicates that the model can **perfectly distinguish  "
                f"between healthy and infected leaves** in real-world "
                f"applications.\n"
                f"- **Training Performance Concerns**: The low AUC on the "
                f"training set might indicate that the model did not  "
                f"generalize well, requiring further investigation.\n"
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

    st.write(f"## Summary & Key Takeaways")

    st.success(
        f"The evaluation of the trained model demonstrates **high classifi- "
        f"cation performance** with strong **generalization to unseen data**. "
        f"The model achieves **high accuracy and low misclassification "
        f"rates**, as shown by the confusion matrices and classification "
        f"reports. The **learning curves indicate effective training**,  "
        f"while the **ROC curve confirmsrobust discrimination** between "
        f"healthy and infected leaves. Moreover, the prediction probability "
        f"histogram highlights the model’s confidence "
        f"in its classifications. These results suggest that the model is "
        f"**well-suited for deployment**, offering a reliable and scalable "
        f"solution for early mildew detection in cherry leaves."
    )

    st.write(
        f"For additional details, visit the "
        f"[Project README](https://github.com/micmic210/mildew-detector/blob/"
        f"main/README.md)."
    )
