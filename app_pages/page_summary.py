import streamlit as st
import glob


def page_summary_body():
    """Displays the project summary with an overview of powdery mildew on cherry trees,
    dataset details, and business requirements.
    """

    # Project Description
    st.write(
        f"## **Project Summary**\n"
        f"Farmy & Foods, a leading agricultural company, faces challenges in managing "
        f"**powdery mildew outbreaks** in its cherry plantations. Currently, mildew detection relies "
        f"on a **manual inspection process**, requiring **30 minutes per tree**, making it **time-consuming "
        f"and impractical** for thousands of trees across multiple farms.\n\n"
        f"To improve efficiency, the **IT and Innovation team** has proposed an **ML-powered detection system** "
        f"that instantly classifies cherry leaves as **healthy or infected** using **image analysis**. "
        f"This solution aims to **reduce inspection time** and could be expanded to other crops if successful. "
        f"The dataset consists of **cherry leaf images** collected from Farmy & Foods' plantations."
    )

    # Auto-Detect Images
    healthy_images = glob.glob("inputs/mildew_dataset/cherry-leaves/train/Healthy/*.JPG")
    infected_images = glob.glob("inputs/mildew_dataset/cherry-leaves/train/Infected/*.JPG")

    if healthy_images and infected_images:
        image_paths = [
            healthy_images[0],
            infected_images[0],
        ]  # Use the first image from each class
        captions = ["Healthy Leaf", "Mildew-Infected Leaf"]

        # Create columns to center images horizontally
        col1, col2, col3 = st.columns(
            [1, 3, 1]
        )  # Adds equal left and right margins for centering

        with col2:  # Inside the center column
            col_left, col_space, col_right = st.columns(
                [1, 0.2, 1]
            )  # Space added between images

            with col_left:
                st.image(image_paths[0], caption=captions[0], width=250)

            with col_right:
                st.image(image_paths[1], caption=captions[1], width=250)

    else:
        st.warning("No images found in dataset! Please check the directory structure.")

    # Business Requirements
    st.write("## Business Requirements")
    st.write(f"""
        **This project addresses three key business requirements:**
        
        - **1. Differentiate** between **healthy cherry leaves** and those affected by powdery mildew.
        - **2. Develop an AI model** to **predict whether a given leaf is healthy or infected.**
        - **3. Generate a detailed prediction report** whenever the mildew detection model is used.  
        This report will provide **insights into the classification results**, offering **transparency and actionable data** for farm management decisions.
    """)

    # Dataset Description
    st.write("### Project Dataset")
    st.write(
        f"""
        **Project Dataset**
        
        The dataset, provided by **Farmy & Foods**, consists of cherry tree leaf images used for training and evaluating the AI model.  
        It includes **over 4,000 labeled images**, sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).  
        To ensure **efficient model training**, a subset of these images is used to balance accuracy and computational cost.
    """
    )

    # External Resources (Styled for Consistency)

    st.info(
        f"""
        **For additional information:**  
        - 📖 [Project README](https://github.com/micmic210/mildew-detector/blob/main/README.md)  
        - 🌎 [Wikipedia: Powdery Mildew](https://en.wikipedia.org/wiki/Powdery_mildew)
        """
    )
