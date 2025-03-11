import streamlit as st
import glob


def page_summary_body():
    """Displays the project summary with an overview of powdery mildew on
    cherry trees, dataset details, and business requirements."""

    # Project Description
    st.write(
        f"## **Project Summary**\n"
        f"Farmy & Foods, a leading agricultural company, faces challenges in "
        f"managing **powdery mildew outbreaks** in its cherry plantations. "
        f"Currently, mildew detection relies on a **manual inspection "
        f"process**, requiring **30 minutes per tree**, making it "
        f"**time-consuming and impractical** for thousands of trees across "
        f"multiple farms.\n\n"
        f"To improve efficiency, the **IT and Innovation team** has proposed "
        f"an **ML-powered detection system** that instantly classifies cherry "
        f"leaves as **healthy or infected** using **image analysis**. "
        f"This solution aims to **reduce inspection time** and could be "
        f"expanded to other crops if successful. The dataset consists of "
        f"**cherry leaf images** collected from Farmy & Foods' plantations."
    )

    # Auto-Detect Images
    healthy_images = glob.glob(
        "inputs/mildew_dataset/cherry-leaves/train/Healthy/*.JPG"
    )
    infected_images = glob.glob(
        "inputs/mildew_dataset/cherry-leaves/train/Infected/*.JPG"
    )

    if healthy_images and infected_images:
        image_paths = [healthy_images[0], infected_images[0]]
        captions = ["Healthy Leaf", "Mildew-Infected Leaf"]

        col1, col2, col3 = st.columns([1, 3, 1])

        with col2:
            col_left, col_space, col_right = st.columns([1, 0.2, 1])

            with col_left:
                st.image(image_paths[0], caption=f"{captions[0]}", width=250)

            with col_right:
                st.image(image_paths[1], caption=f"{captions[1]}", width=250)

    else:
        st.warning(f"No images found in dataset! Please check the directory.")

    # Business Requirements
    st.write(f"## Business Requirements")
    st.write(
        f"**This project addresses three key business requirements:**\n\n"
        f"- **1. Differentiate** between **healthy cherry leaves** and those "
        f"affected by powdery mildew.\n"
        f"- **2. Develop an AI model** to **predict whether a given leaf is "
        f"healthy or infected.**\n"
        f"- **3. Generate a detailed prediction report** whenever the "
        f"mildew detection model is used. This report will provide "
        f"**insights into the classification results**, offering "
        f"**transparency and actionable data** for farm management."
    )

    # Dataset Description
    st.write(f"### Project Dataset")
    st.write(
        f"**Project Dataset**\n\n"
        f"The dataset, provided by **Farmy & Foods**, consists of "
        f"cherry tree leaf images used for training and evaluating the "
        f"AI model. It includes **over 4,000 labeled images**, sourced from "
        f"[Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).\n"
        f"To ensure **efficient model training**, a subset of these images "
        f"is used to balance accuracy and computational cost."
    )

    # External Resources
    st.info(
        f"**For additional information:**\n"
        f"- 📖 [Project README](https://github.com/micmic210/mildew-detector/"
        f"blob/main/README.md)\n"
        f"- 🌎 [Wikipedia: Powdery Mildew](https://en.wikipedia.org/wiki/"
        f"Powdery_mildew)"
    )
