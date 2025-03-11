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

# Apply Global CSS Styles for Font & Layout
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@
    300;400;700&display=swap');

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

    /* Sidebar: Adjust width for better responsiveness */
    @media screen and (max-width: 1024px) {
        .stSidebar {
            max-width: 180px !important;
        }
    }

    /* HIDE SIDEBAR AND SHOW HAMBURGER MENU */
    @media screen and (max-width: 768px) {
        .stSidebar {
            display: none !important;
        }
        .hamburger-menu {
            display: block !important;
        }
    }

    @media screen and (min-width: 769px) {
        .hamburger-menu {
            display: none !important;
        }
    }

    /* Hamburger button styling */
    .hamburger-menu {
        position: fixed;
        top: 15px;
        left: 15px;
        z-index: 1000;
        background-color: #567d46;
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hamburger Menu Button (Simulated Sidebar Toggle)
st.markdown(
    """
    <script>
    function toggleSidebar() {
        var sidebar = document.querySelector('.stSidebar');
        if (sidebar.style.display === "none") {
            sidebar.style.display = "block";
        } else {
            sidebar.style.display = "none";
        }
    }
    </script>

    <button class="hamburger-menu" onclick="toggleSidebar()">
        ☰ Menu
    </button>
    """,
    unsafe_allow_html=True,
)

# Run the Multi-Page App
app.run()
