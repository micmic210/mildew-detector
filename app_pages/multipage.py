import streamlit as st


class MultiPage:
    def __init__(self, app_name) -> None:
        """
        Initializes the multi-page application.
        """
        self.pages = []
        self.app_name = app_name

        # Streamlit page configuration
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

        # Title & Styling
        st.markdown(
            """
            <style>
                .main-title {
                    text-align: center;
                    font-weight: bold;
                    color: #567d46;
                    margin-top: 0px;
                    overflow: hidden;
                    white-space: nowrap; /* Prevent title from breaking */
                    text-overflow: ellipsis;
                }

                @media screen and (min-width: 1025px) {
                    .main-title {
                        font-size: 55px;
                    }
                }

                @media screen and (max-width: 1024px) {
                    .main-title {
                        font-size: 45px;
                    }
                }

                @media screen and (max-width: 768px) {
                    .main-title {
                        font-size: 38px;
                    }
                }

                @media screen and (max-width: 480px) {
                    .main-title {
                        font-size: 32px;
                        white-space: normal;
                    }
                }
                h2 {
                    font-size: 28px;
                }

                p {
                    font-size: 18px;
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

        # Sidebar Navigation with Hamburger Menu for Mobile
        with st.sidebar:
            page = st.radio(
                "Menu",
                self.pages,
                format_func=lambda page: page["title"]
            )

        # Render the selected page
        page["function"]()
