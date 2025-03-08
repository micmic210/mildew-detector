import streamlit as st
import glob


def page_summary_body():
    """Displays the project summary with an overview of powdery mildew on cherry trees,
    dataset details, and business requirements.
    """

    # Page Title
    st.write("## Project Summary")
    st.write("### General Information")
    st.write("#### Powdery Mildew on Cherry Trees")

    # Project Description
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

    # Auto-Detect Images
    healthy_images = glob.glob(
        "inputs/mildew_dataset/cherry-leaves/train/Healthy/*.JPG"
    )
    infected_images = glob.glob(
        "inputs/mildew_dataset/cherry-leaves/train/Infected/*.JPG"
    )

    if healthy_images and infected_images:
        image_paths = [
            healthy_images[0],
            infected_images[0],
        ]  # Use the first image from each class
        captions = ["Healthy Leaf", "Mildew-Infected Leaf"]

        # Use columns for responsiveness (side by side on wide screens, stacked on small screens)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image_paths[0], caption=captions[0], width=250)
        with col2:
            st.image(image_paths[1], caption=captions[1], width=250)

    else:
        st.warning("No images found in dataset! Please check the directory structure.")

    # External Resources
    st.info(
        "**For additional information:**\n"
        "- 📖 [Project README](https://github.com/micmic210/mildew-detector/blob/main/README.md)\n"
        "- 🌎 [Wikipedia: Powdery Mildew](https://en.wikipedia.org/wiki/Powdery_mildew)"
    )

    # Business Requirements
    st.write("## Business Requirements")
    st.write(
        "This project addresses **two key business requirements:**\n"
        "1️⃣ **Differentiate** between **healthy cherry leaves** and those affected by powdery mildew.\n"
        "2️⃣ **Develop an AI model** to **predict whether a given leaf is healthy or infected.**"
    )

    # Dataset Description
    st.write("### Project Dataset")
    st.write(
        "The dataset, provided by **Farmy & Foods**, consists of cherry tree leaf images used for "
        "training and evaluating the AI model.\n\n"
        "It includes **over 4,000 labeled images**, sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). "
        "For **efficient model training**, a subset of these images is used to balance accuracy and computational cost."
    )
