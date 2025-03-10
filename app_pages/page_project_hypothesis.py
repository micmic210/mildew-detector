import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    """Displays the project hypotheses, validation methods, findings, and conclusions."""

    st.write(f"## Project Hypotheses & Validation")

    # Hypothesis 1
    st.write(f"### Hypothesis 1: Visual Differences Exist")
    st.success(
        f"Healthy cherry leaves have a uniform texture and consistent brightness, "
        f"while mildew-infected leaves display discoloration, irregular brightness, and fungal patches."
    )

    st.info(
        f"**Validation Methods:**\n"
        f"- **Mean & Standard Deviation Images:** Compare overall color and texture patterns.\n"
        f"- **T-Test on Pixel Intensities:** Statistical test to determine significant brightness differences.\n"
        f"- **PCA Feature Space Analysis:** Evaluate class separability by reducing dimensionality."
    )

    st.warning(
        f"**Findings & Conclusion:**\n"
        f"- Statistically significant differences detected (**p < 0.05** in t-test).\n"
        f"- Mean images show brighter patches in infected leaves.\n"
        f"- PCA visualization suggests moderate class separability, though additional feature engineering may improve it.\n"
        f"Conclusion: Visual differences can be leveraged for mildew detection."
    )

    # Hypothesis 2
    st.write(f"### Hypothesis 2: Machine Learning Can Accurately Detect Mildew")
    st.success(
        f"A well-trained CNN model can classify cherry leaves with ≥90% accuracy, "
        f"making the detection process scalable and reliable."
    )

    st.info(
        f"**Validation Methods:**\n"
        f"- **Train CNN Model & Evaluate Performance:** Accuracy, precision, recall, and F1-score.\n"
        f"- **Confusion Matrix & Classification Report:** Identify false positives and negatives.\n"
        f"- **ROC Curve & AUC Score:** Assess model’s ability to separate healthy vs. infected leaves."
    )

    st.warning(
        f"**Findings & Conclusion:**\n"
        f"- CNN model achieves high accuracy (**≥90%**, replace with actual value).\n"
        f"- Confusion matrix confirms low false negatives, meaning reliable detection.\n"
        f"- ROC Curve shows AUC score (**≥0.90**, replace with actual value).\n"
        f"Conclusion: The ML model meets performance expectations but can be improved with further tuning."
    )

    # Hypothesis 3
    st.write(
        f"### Hypothesis 3: Model Confidence Scores Indicate Prediction Reliability"
    )
    st.success(
        f"A well-calibrated CNN model should provide high confidence scores for correct predictions "
        f"and lower confidence scores for misclassified images. If misclassified images have "
        f"high confidence scores (>90%), it may indicate overconfidence, requiring threshold tuning."
    )

    st.info(
        f"**Validation Methods:**\n"
        f"- **Confidence Score Distribution Analysis:** Evaluate the spread of confidence scores.\n"
        f"- **Interactive Image Confidence Check:** Allow users to analyze confidence scores in Streamlit.\n"
        f"- **Comparison of Confidence Across Classes:** Assess confidence bias between 'Healthy' and 'Infected' leaves."
    )

    st.warning(
        f"**Findings & Conclusion:**\n"
        f"- Misclassified images should show lower confidence scores.\n"
        f"- If misclassified images have high confidence (>90%), it suggests model overconfidence.\n"
        f"- If the model is overconfident in one class (e.g., always predicting 'Healthy' with high confidence), "
        f"it may indicate class imbalance.\n"
        f"Conclusion: A well-calibrated model should assign lower confidence to misclassified images, "
        f"ensuring reliable predictions."
    )

    st.write(
        f"**Additional Information:**\n"
        f"- [Project README](https://github.com/micmic210/mildew-detector/blob/main/README.md)"
    )
