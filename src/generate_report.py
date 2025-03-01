import joblib
import os
import glob
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def find_latest_model():
    draft_dirs = sorted(glob.glob("outputs/draft_*"), key=os.path.getmtime, reverse=True)
    return draft_dirs[0] if draft_dirs else None

def generate_pdf(model_path):
    file_path = os.path.join(model_path, "evaluation.pkl")

    if not os.path.exists(file_path):
        print(f"No evaluation file found in {model_path}. Run the notebook first.")
        return

    results = joblib.load(file_path)
    pdf_path = os.path.join(model_path, "trial_report.pdf")

    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, f"Model Evaluation Report - {model_path}")

    c.setFont("Helvetica", 11)
    y = 720
    c.drawString(50, y, f"Accuracy: {round(results.get('test_accuracy', 0), 3)}")
    y -= 20
    c.drawString(50, y, f"Validation Accuracy: {round(results.get('val_accuracy', 0), 3)}")
    y -= 20

    c.save()
    print(f"PDF Generated: {pdf_path}")

if __name__ == "__main__":
    latest_model = find_latest_model()
    if latest_model:
        print(f"Using latest model: {latest_model}")
        generate_pdf(latest_model)
    else:
        print("No model found in outputs/")