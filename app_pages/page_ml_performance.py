import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from matplotlib.image import imread


# Function to safely load test evaluation from evaluation.pkl
def load_test_evaluation(version):
    file_path = f"outputs/{version}/evaluation.pkl"
    try:
        with open(file_path, "rb") as file:
            test_eval = pickle.load(file)

        # Ensure data is in expected format
        if isinstance(test_eval, dict):  # If dict, extract values
            test_eval = list(test_eval.values())
        return test_eval  # Expecting [loss, accuracy]

    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading evaluation.pkl: {e}")
        return None


# Define the main function to display model performance metrics
def page_ml_performance_metrics():
    """Displays the performance metrics of the trained ML model."""

    version = "v1"

    st.write("## Model Performance & Evaluation")

    st.info(
        "This section presents how the dataset was split for training, "
        "how well the model performed, and key performance metrics."
    )

    # Dataset Split & Class Distribution
    st.write("### Dataset Split & Class Distribution")

    labels_distribution = plt.imread(f"outputs/{version}/labels_distribution.png")
    st.image(
        labels_distribution,
        caption="Class Distribution in Train, Validation, and Test Sets",
    )

    st.warning(
        "**Dataset Partitioning:**\n"
        "- **Training Set (70%)** → Model learns from this set.\n"
        "- **Validation Set (10%)** → Used for model fine-tuning.\n"
        "- **Test Set (20%)** → Evaluates final model performance on unseen data."
    )

    st.write("---")

    # Classification Reports (Train & Test)
    st.write("### Classification Report (Train vs. Test)")

    # Load the classification reports
    train_report_path = f"outputs/{version}/classification_report_train.csv"
    test_report_path = f"outputs/{version}/classification_report_test.csv"

    # Function to apply color formatting to DataFrame
    def highlight_cells(val):
        if isinstance(val, (int, float)):  # Ensure we only format numbers
            if val >= 0.9:
                return "background-color: #c6efce; color: #006400;"  # Green for high values
            elif val <= 0.6:
                return (
                    "background-color: #ffc7ce; color: #9c0006;"  # Red for low values
                )
        return ""

    # Display Train Set Classification Report
    st.write("#### Train Set Classification Report")
    train_report = pd.read_csv(train_report_path)
    st.dataframe(train_report.style.applymap(highlight_cells))

    st.write("---")  # Add spacing between reports

    # Display Test Set Classification Report
    st.write("#### Test Set Classification Report")
    test_report = pd.read_csv(test_report_path)
    st.dataframe(test_report.style.applymap(highlight_cells))

    st.warning(
        "**Classification Report Interpretation:**\n"
        "- **Train Set:** Measures how well the model classifies training data.\n"
        "- **Test Set:** Evaluates model's generalization to new, unseen data.\n"
        "- **Precision:** Accuracy of positive class predictions.\n"
        "- **Recall:** Ability to detect actual positive cases.\n"
        "- **F1 Score:** Balance of Precision & Recall (ideal when close to 1.0).\n"
        "- **A well-generalized model** should show **similar performance on both Train & Test sets**."
    )

    st.write("---")

    # Confusion Matrix (Train vs. Test)
    st.write("### Confusion Matrix (Train vs. Test)")

    # Define file paths
    train_cm_path = f"outputs/{version}/confusion_matrix_train.png"
    test_cm_path = f"outputs/{version}/confusion_matrix_test.png"

    # Load and display Train Confusion Matrix
    train_cm = plt.imread(train_cm_path)
    st.image(train_cm, caption="Confusion Matrix (Train Set)", use_column_width=True)

    # Load and display Test Confusion Matrix
    test_cm = plt.imread(test_cm_path)
    st.image(test_cm, caption="Confusion Matrix (Test Set)", use_column_width=True)

    # Explanation of Confusion Matrix
    st.warning(
        "**Confusion Matrix Interpretation:**\n"
        "- **True Positives (TP) & True Negatives (TN):** Correctly classified cases.\n"
        "- **False Positives (FP) & False Negatives (FN):** Incorrect classifications.\n"
        "- A good model should have **low False Positives & False Negatives**, while maintaining high TP & TN."
    )

    st.write("---")

    # ROC Curve
    st.write("### ROC Curve")

    model_roc = plt.imread(f"outputs/{version}/roc_curve.png")
    st.image(model_roc, caption="ROC Curve")

    st.warning(
        "**ROC Curve Analysis:**\n"
        "- **True Positive Rate (TPR):** Correctly classified positives.\n"
        "- **False Positive Rate (FPR):** Incorrectly classified negatives.\n"
        "- **AUC Score (≥0.90 recommended):** Measures the model’s ability to distinguish between classes."
    )

    st.write("---")

    # Model Learning Curves (Accuracy & Loss)
    st.write("### Model Training Performance")

    col1, col2 = st.columns(2)

    with col1:
        model_acc = plt.imread(f"outputs/{version}/accuracy_curve_softmax.png")
        st.image(model_acc, caption="Model Training Accuracy")

    with col2:
        model_loss = plt.imread(f"outputs/{version}/loss_curve_softmax.png")
        st.image(model_loss, caption="Model Training Loss")

    st.warning(
        "**Training & Validation Curves:**\n"
        "- **Loss Curve:** Measures how well the model learns over time.\n"
        "- **Accuracy Curve:** Evaluates model performance across training epochs.\n"
        "- **Ideal Scenario:** Validation accuracy should closely follow training accuracy to avoid overfitting."
    )

    st.write("---")

    # Histograms
    st.write("### Histogram of Prediction Probabilities (Test Set)")

    model_hist = plt.imread(f"outputs/{version}/histogram_test.png")
    st.image(model_hist, caption="Prediction Probabilities Histogram (Test Set)")

    st.warning(
        "**Histogram Analysis:**\n"
        "- Displays confidence score distribution for predictions.\n"
        "- Well-calibrated models should have **high-confidence correct predictions** "
        "and **low-confidence incorrect predictions**."
    )

    st.write("---")

    # Generalized Model Performance on Test Set
    st.write("### Final Model Performance on Test Set")

    test_eval = load_test_evaluation(version)

    # Ensure test_eval is correctly formatted
    if isinstance(test_eval, (list, tuple)) and len(test_eval) == 2:
        df_test_eval = pd.DataFrame(
            {"Metric": ["Loss", "Accuracy"], "Value": test_eval}
        )
        st.table(df_test_eval)
    else:
        st.error("Unexpected format in evaluation.pkl. Verify file contents.")

    st.write(
        "For additional details, visit the "
        "[Project README](https://github.com/micmic210/mildew-detector/blob/main/README.md)."
    )
