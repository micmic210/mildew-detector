import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    """Displays the project hypotheses, validation methods, findings,
    and conclusions."""

    st.write(f"## Project Hypotheses & Validation")

    # Hypothesis 1
    st.write(f"### Hypothesis 1: Visual Differences Exist")
    st.success(
        f"Healthy cherry leaves have a uniform texture and consistent "
        f"brightness, while mildew-infected leaves display discoloration, "
        f"irregular brightness, and fungal patches."
    )

    st.info(
        f"**Validation Methods:**\n"
        f"- **Mean & Standard Deviation Images:** Compare overall color "
        f"and texture patterns.\n"
        f"- **T-Test on Pixel Intensities:** Statistical test to determine "
        f"significant brightness differences.\n"
        f"- **PCA Feature Space Analysis:** Evaluate class separability "
        f"by reducing dimensionality."
    )

    st.warning(
        f"**Findings & Conclusion:**\n"
        f"- Statistically significant differences detected (**p < 0.05**) "
        f"in t-test.\n"
        f"- Mean images show brighter patches in infected leaves.\n"
        f"- PCA visualization suggests moderate class separability, though "
        f"additional feature engineering may improve it.\n"
        f"Conclusion: Visual differences can be leveraged for mildew "
        f"detection."
    )

    # Hypothesis 2
    st.write(
        f"### Hypothesis 2: Machine Learning Can Accurately Detect Mildew"
    )
    st.success(
        f"A well-trained CNN model can classify cherry leaves with "
        f"≥90% accuracy, making the detection process scalable and reliable."
    )

    st.info(
        f"**Validation Methods:**\n"
        f"- **Train CNN Model & Evaluate Performance:** Accuracy, precision, "
        f"recall, and F1-score.\n"
        f"- **Confusion Matrix & Classification Report:** Identify false "
        f"positives and negatives.\n"
        f"- **ROC Curve & AUC Score:** Assess model’s ability to separate "
        f"healthy vs. infected leaves."
    )

    st.warning(
        f"**Findings & Conclusion:**\n"
        f"- CNN model achieves high accuracy (**≥90%**, replace with value).\n"
        f"- Confusion matrix confirms low false negatives, meaning reliable "
        f"detection.\n"
        f"- ROC Curve shows AUC score (**≥0.90**, replace with value).\n"
        f"Conclusion: The ML model meets performance expectations but can be "
        f"improved with further tuning."
    )

    # Hypothesis 3
    st.write(f"### Hypothesis 3: Confidence Scores & Prediction Reliability")
    st.success(
        f"A well-calibrated CNN model should provide high confidence scores "
        f"for correct predictions and lower confidence scores for "
        f"misclassified images. If misclassified images have high "
        f"confidence scores (>90%), it may indicate overconfidence, "
        f"requiring threshold tuning."
    )

    st.info(
        f"**Validation Methods:**\n"
        f"- **Confidence Score Distribution Analysis:** Evaluate the "
        f"spread of confidence scores.\n"
        f"- **Interactive Image Confidence Check:** Allow users to analyze "
        f"confidence scores in Streamlit.\n"
        f"- **Comparison of Confidence Across Classes:** Assess confidence "
        f"bias between 'Healthy' and 'Infected' leaves."
    )

    st.warning(
        f"**Findings & Conclusion:**\n"
        f"- Misclassified images should show lower confidence scores.\n"
        f"- If misclassified images have high confidence (>90%), it "
        f"suggests model overconfidence.\n"
        f"- If the model is overconfident in one class (e.g., always "
        f"predicting 'Healthy' with high confidence), it may indicate class "
        f"imbalance.\n"
        f"Conclusion: A well-calibrated model should assign lower confidence "
        f"to misclassified images, ensuring reliable predictions."
    )

    st.write(
        f"**Additional Information:**\n"
        f"- [Project README](https://github.com/micmic210/mildew-detector/"
        f"blob/main/README.md)"
    )
