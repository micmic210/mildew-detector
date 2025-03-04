import streamlit as st

# Define the layout and project summary page


def page_summary_body():
    """Displays the project summary with an overview of powdery mildew on cherry trees,
    dataset details, and business requirements.
    """

    st.write("## 🍒 Project Summary")
    st.write("### 🌿 General Information")
    st.write("#### Powdery Mildew on Cherry Trees")

    st.write(
        "Powdery mildew is a widespread fungal disease that affects a variety of plants, "
        "manifesting as **light grey or white powdery spots** primarily on leaves but also "
        "on stems, flowers, and fruits. These spots gradually spread, covering entire leaves, "
        "particularly targeting new plant growth.\n\n"
        "While not typically fatal, **untreated powdery mildew can severely impact plant health** "
        "by restricting access to water and nutrients. Common symptoms include:\n"
        "- **Yellowing and curling leaves**\n"
        "- **Weakened plant growth**\n"
        "- **Reduced blooming and slowed development**\n\n"
        "Currently, manual inspection of each **cherry tree** takes approximately **30 minutes per tree**, "
        "with an **additional minute for treatment** when necessary. Given the vast number of trees "
        "across multiple farms, this **manual approach is impractical**. To streamline the process, "
        "the IT team has proposed a **machine learning (ML) system** capable of detecting powdery mildew "
        "instantly using **leaf images**. If successful, this technology could be expanded to **other crops**."
    )

    # Display healthy and infected leaf images
    image_paths = [
        "inputs/cherry-leaves_dataset/cherry-leaves/healthy/00a8e886-d172-4261-85e2-780b3c50ad4d___JR_HL 4156.JPG",
        "inputs/cherry-leaves_dataset/cherry-leaves/fungal-infected/0a283423-3a6d-43a4-92e5-267c8153ca45___FREC_Pwd.M 4921_flipLR.JPG",
    ]
    captions = ["Healthy Leaf", "Mildew-Infected Leaf"]

    st.image(image_paths, caption=captions, width=300)

    st.info(
        "**For additional information:**\n"
        "- 📖 [Project README](https://github.com/micmic210/mildew-detector/blob/main/README.md)\n"
        "- 🌎 [Wikipedia: Powdery Mildew](https://en.wikipedia.org/wiki/Powdery_mildew)"
    )

    st.write("## 📌 Business Requirements")
    st.write(
        "This project addresses **two key business requirements:**\n"
        "**Differentiate** between **healthy cherry leaves** and those affected by powdery mildew.\n"
        "**Develop an AI model** to **predict whether a given leaf is healthy or infected.**"
    )

    st.write("### 📊 Project Dataset")
    st.write(
        "The dataset, provided by **Farmy & Foods**, consists of cherry tree leaf images used for "
        "training and evaluating the AI model.\n\n"
        "It includes **over 4,000 labeled images**, sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). "
        "For **efficient model training**, a subset of these images is used to balance accuracy and computational cost."
    )
