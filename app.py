import streamlit as st
from app_pages.multipage import MultiPage

# Load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_leaves_visualizer import page_leaves_visualizer_body
from app_pages.page_mildew_detector import page_mildew_detector_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_ml_performance import page_ml_performance_metrics

# Initialize the multi-page app
app = MultiPage(app_name="Powdery Mildew Detector")

# Add pages to the app
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Leaves Visualizer", page_leaves_visualizer_body)
app.add_page("Mildew Detector", page_mildew_detector_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("ML Performance Metrics", page_ml_performance_metrics)

# Apply Roboto font globally
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* Make the main content full width on mobile */
    @media screen and (max-width: 768px) {
        .stApp {
            padding: 10px;
        }
        .block-container {
            padding: 0rem 1rem;
        }
    }

    /* Fix sidebar width on small screens */
    @media screen and (max-width: 1024px) {
        .stSidebar {
            width: 250px !important;
        }
    }

    /* Improve responsiveness of tables */
    table {
        width: 100% !important;
    }

    /* Resize images dynamically */
    img {
        max-width: 100% !important;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Run the app
app.run()
