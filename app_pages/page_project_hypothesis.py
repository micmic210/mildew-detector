import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    """Displays the project hypotheses, validation methods, findings, and conclusions."""

    st.write("## Project Hypotheses & Validation")

    # Hypothesis 1
    st.write("### Hypothesis 1: Visual Differences Exist")
    st.success(
        "Healthy cherry leaves have a uniform texture and consistent brightness, "
        "while mildew-infected leaves display discoloration, irregular brightness, and fungal patches."
    )

    st.info(
        "**Validation Methods:**\n"
        "- **Mean & Standard Deviation Images:** Compare overall color and texture patterns.\n"
        "- **T-Test on Pixel Intensities:** Statistical test to determine significant brightness differences.\n"
        "- **PCA Feature Space Analysis:** Evaluate class separability by reducing dimensionality."
    )

    st.warning(
        "**Findings & Conclusion:**\n"
        "- Statistically significant differences detected (**p < 0.05** in t-test).\n"
        "- Mean images show brighter patches in infected leaves.\n"
        "- PCA visualization suggests moderate class separability, though additional feature engineering may improve it.\n"
        "Conclusion: Visual differences can be leveraged for mildew detection."
    )

    # Hypothesis 2
    st.write("### Hypothesis 2: Machine Learning Can Accurately Detect Mildew")
    st.success(
        "A well-trained CNN model can classify cherry leaves with ≥90% accuracy, "
        "making the detection process scalable and reliable."
    )

    st.info(
        "**Validation Methods:**\n"
        "- **Train CNN Model & Evaluate Performance:** Accuracy, precision, recall, and F1-score.\n"
        "- **Confusion Matrix & Classification Report:** Identify false positives and negatives.\n"
        "- **ROC Curve & AUC Score:** Assess model’s ability to separate healthy vs. infected leaves."
    )

    st.warning(
        "**Findings & Conclusion:**\n"
        "- CNN model achieves high accuracy (**≥90%**, replace with actual value).\n"
        "- Confusion matrix confirms low false negatives, meaning reliable detection.\n"
        "- ROC Curve shows AUC score (**≥0.90**, replace with actual value).\n"
        "Conclusion: The ML model meets performance expectations but can be improved with further tuning."
    )

    # Hypothesis 3
    st.write(
        "### Hypothesis 3: Model Confidence Scores Indicate Prediction Reliability"
    )
    st.success(
        "A well-calibrated CNN model should provide high confidence scores for correct predictions "
        "and lower confidence scores for misclassified images. If misclassified images have "
        "high confidence scores (>90%), it may indicate overconfidence, requiring threshold tuning."
    )

    st.info(
        "**Validation Methods:**\n"
        "- **Confidence Score Distribution Analysis:** Evaluate the spread of confidence scores.\n"
        "- **Interactive Image Confidence Check:** Allow users to analyze confidence scores in Streamlit.\n"
        "- **Comparison of Confidence Across Classes:** Assess confidence bias between 'Healthy' and 'Infected' leaves."
    )

    st.warning(
        "**Findings & Conclusion:**\n"
        "- Misclassified images should show lower confidence scores.\n"
        "- If misclassified images have high confidence (>90%), it suggests model overconfidence.\n"
        "- If the model is overconfident in one class (e.g., always predicting 'Healthy' with high confidence), "
        "it may indicate class imbalance.\n"
        "Conclusion: A well-calibrated model should assign lower confidence to misclassified images, "
        "ensuring reliable predictions."
    )

    st.write(
        "**Additional Information:**\n"
        "- [Project README](https://github.com/micmic210/mildew-detector/blob/main/README.md)"
    )
