import numpy as np
import pandas as pd
import os
import base64
from datetime import datetime
import joblib


def download_dataframe_as_csv(df):
    """
    Convert a DataFrame to a CSV file and provide a download link.
    """

    datetime_now = datetime.now().strftime("%d%b%Y_%Hh%Mmin%Ss")
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = (
        f'<a href="data:file/csv;base64,{b64}" '
        f'download="Report_{datetime_now}.csv" target="_blank">'
        f"Download Report</a>"
    )
    return href


def load_pkl_file(file_path):
    """
    Load a serialized object from a pickle file.
    """
    return joblib.load(filename=file_path)
