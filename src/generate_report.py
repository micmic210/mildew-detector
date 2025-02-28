import joblib
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate_pdf():
    file_path = "outputs/draft/evaluation.pkl"

    if os.path.exists(file_path):
        results = joblib.load(file_path)
        pdf_path = "outputs/draft/trial_report.pdf"

        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 750, "Model Evaluation Report")

        c.setFont("Helvetica", 11)
        y = 720

        for key, value in results.items():
            if isinstance(value, dict):
                c.drawString(50, y, f"{key}:")
                y -= 20
                for sub_key, sub_value in value.items():
                    c.drawString(70, y, f"{sub_key}: {round(sub_value, 3)}")
                    y -= 15
            else:
                c.drawString(50, y, f"{key}: {round(value, 3)}")
                y -= 20

            if y < 100:
                c.showPage()
                y = 750

        c.save()
        print(f"PDF Generated: {pdf_path}")
    else:
        print("No evaluation file found. Run the notebook first.")


if __name__ == "__main__":
    generate_pdf()
