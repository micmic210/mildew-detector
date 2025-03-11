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

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
    }

    /* Ensure main content adjusts properly */
    .stApp {
        padding: 5px;
    }

    .block-container {
        padding: 0.5rem 0.8rem;
    }

    /* Sidebar: Reduce width for better responsiveness */
    @media screen and (max-width: 1024px) {
        .stSidebar {
            max-width: 180px !important;
        }
    }

    @media screen and (max-width: 768px) {
        .stSidebar {
            max-width: 150px !important;
        }
    }

    @media screen and (max-width: 480px) {
        .stSidebar {
            display: none; /* Hide sidebar on small screens */
        }
    }

    /* Tables should remain scrollable */
    table {
        width: 100% !important;
        display: block;
        overflow-x: auto;
        white-space: nowrap;
    }

    /* Make images responsive */
    img {
        max-width: 100% !important;
        height: auto;
        display: block;
        margin: 0 auto;
    }

    /* Prevent text overflow */
    .stMarkdown, .stTextInput, .stTextArea, .stSelectbox {
        word-wrap: break-word !important;
        white-space: normal !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
app.run()
