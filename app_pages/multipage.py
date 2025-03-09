import streamlit as st


class MultiPage:
    def __init__(self, app_name) -> None:
        """
        Initializes the multi-page application.
        """
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🍃",
            layout="wide",
        )

    def add_page(self, title, func) -> None:
        """
        Adds a new page to the multi-page app.

        Parameters:
        - title (str): The name of the page.
        - func (function): The function that renders the page.
        """
        self.pages.append({"title": title, "function": func})

    def run(self):
        """
        Runs the multi-page app with the default Streamlit sidebar.
        """
        # **Move Title Up**
        st.markdown(
            """
            <style>
                .main-title {
                    text-align: center;
                    font-size: 50px;
                    font-weight: bold;
                    color: #567d46;
                    margin-top: -50px; /* Move title up */
                }
                .separator {
                    border: 2px solid #567d46;
                    margin-bottom: 20px;
                }
            </style>

            <h1 class='main-title'>🍃 Powdery Mildew Detector</h1>
            <hr class='separator'>
            """,
            unsafe_allow_html=True,
        )

        # Sidebar Navigation
        page = st.sidebar.radio(
            "Menu", self.pages, format_func=lambda page: page["title"]
        )

        # Render the selected page
        page["function"]()
