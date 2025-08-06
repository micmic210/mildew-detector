import streamlit as st
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import itertools
import random


def page_leaves_visualizer_body():
    """Displays visualizations to differentiate Healthy vs. Mildew-Infected
    leaves."""

    st.write("## Leaves Visualizer")
    st.info(
        f"This study visually differentiates **Healthy vs. Mildew-Infected** "
        f"cherry leaves, helping to understand patterns before applying "
        f"machine learning."
    )

    version = "v1"

    avg_healthy_path = f"outputs/{version}/avg_var_Healthy.png"
    avg_infected_path = f"outputs/{version}/avg_var_Infected.png"
    diff_avg_path = f"outputs/{version}/avg_diff.png"

    if st.checkbox("Average & Variability Images"):
        if os.path.exists(avg_healthy_path) and os.path.exists(avg_infected_path):
            avg_healthy = imread(avg_healthy_path)
            avg_infected = imread(avg_infected_path)

            st.warning(
                f"The **average and variability images** do not reveal "
                f"obvious patterns that allow intuitive differentiation. "
                f"However, mildew-infected leaves display more **white "
                f"streaks at the center**, suggesting some textural "
                f"differences."
            )

            st.image(
                [avg_healthy, avg_infected],
                caption=[
                    f"Healthy Leaf - Average & Variability",
                    f"Mildew-Infected Leaf - Average & Variability",
                ],
            )
        else:
            st.error(
                f"⚠️ Required visualization files not found. Please check "
                f"your preprocessing steps."
            )

        st.write("---")

    if st.checkbox("Difference Between Average Healthy & Infected Leaves"):
        if os.path.exists(diff_avg_path):
            diff_avg = imread(diff_avg_path)

            st.warning(
                f"The **difference between average images** does not provide "
                f"strong visual cues for classification. Further feature "
                f"extraction techniques may be required."
            )

            st.image(diff_avg, caption="Difference Between Average Images")
        else:
            st.error(
                f"⚠️ Difference image not found. Ensure it's generated in "
                f"preprocessing."
            )

        st.write("---")

    if st.checkbox("Image Montage"):
        st.write("To refresh the montage, click on 'Create Montage'.")

        my_data_dir = "inputs/mildew_dataset/cherry-leaves"
        labels = os.listdir(os.path.join(my_data_dir, "validation"))
        label_to_display = st.selectbox("Select Class Label", options=labels, index=0)

        if st.button("Create Montage"):
            image_montage(
                dir_path=f"{my_data_dir}/validation",
                label_to_display=label_to_display,
                nrows=8,
                ncols=3,
                figsize=(10, 25),
            )

        st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(10, 10)):
    """Generates a montage of images for a selected class label."""

    sns.set_style("white")
    labels = os.listdir(dir_path)

    if label_to_display in labels:
        images_list = os.listdir(os.path.join(dir_path, label_to_display))

        if nrows * ncols > len(images_list):
            st.warning(
                f"Not enough images. Available: {len(images_list)}, "
                f"requested: {nrows * ncols}. Adjust nrows/ncols."
            )
            return

        img_idx = random.sample(images_list, nrows * ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        plot_idx = list(itertools.product(range(nrows), range(ncols)))

        for i, img_name in enumerate(img_idx):
            img_path = os.path.join(dir_path, label_to_display, img_name)
            img = imread(img_path)
            img_shape = img.shape
            row, col = plot_idx[i]

            axes[row, col].imshow(img)
            axes[row, col].set_title(f"{img_shape[1]}px x {img_shape[0]}px")
            axes[row, col].axis("off")

        plt.tight_layout()
        st.pyplot(fig=fig)
    else:
        st.error(
            f"⚠️ The selected label '{label_to_display}' does not exist. "
            f"Please choose from available options."
        )
