import streamlit as st


class MultiPage:
    """
    A class to manage multiple pages in a Streamlit app.
    """

    def __init__(self, app_name: str) -> None:
        """
        Initializes the multipage application.

        Parameters:
        - app_name (str): The title of the application.
        """
        self.pages = []
        self.app_name = app_name

        # Set the page configuration
        st.set_page_config(
            page_title=self.app_name,
            page_icon="🍒", 
            layout="wide",  # "centered" is another option
        )

    def add_page(self, title: str, func) -> None:
        """
        Adds a new page to the multipage app.

        Parameters:
        - title (str): The name of the page.
        - func (function): The function that renders the page.
        """
        self.pages.append({"title": title, "function": func})

    def run(self) -> None:
        """
        Runs the multipage app by displaying the selected page.
        """
        # Set the sidebar layout
        st.sidebar.header("📌 Navigation")
        page = st.sidebar.radio(
            "Select a page:", self.pages, format_func=lambda page: page["title"]
        )
        page["function"]()
