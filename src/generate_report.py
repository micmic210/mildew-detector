import joblib
import os
import glob
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def find_latest_model():
    draft_dirs = sorted(
        glob.glob("outputs/draft_*"), key=os.path.getmtime, reverse=True
    )
    return draft_dirs[0] if draft_dirs else None


def generate_pdf(model_path):
    file_path = os.path.join(model_path, "evaluation.pkl")

    if not os.path.exists(file_path):
        print(f"No evaluation file found in {model_path}. Run the notebook first.")
        return

    results = joblib.load(file_path)
    pdf_path = os.path.join(model_path, "trial_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Set Font
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, f"Model Evaluation Report - {model_path}")

    # Accuracy & Validation Accuracy
    c.setFont("Helvetica", 11)
    y = 720
    c.drawString(50, y, f"Accuracy: {round(results.get('test_accuracy', 0), 3)}")
    y -= 20
    c.drawString(
        50, y, f"Validation Accuracy: {round(results.get('val_accuracy', 0), 3)}"
    )
    y -= 30

    # Model Summary
    model_summary_path = os.path.join(model_path, "model_summary.txt")
    if os.path.exists(model_summary_path):
        with open(model_summary_path, "r") as f:
            model_summary = f.readlines()
    else:
        model_summary = ["No model summary available"]

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Model Summary:")
    y -= 20
    c.setFont("Helvetica", 9)

    for line in model_summary[:10]:  # Limiting lines to avoid overflow
        c.drawString(50, y, line.strip())
        y -= 12
        if y < 100:
            c.showPage()
            y = 750

    # Classification Report
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Classification Report:")
    y -= 20
    c.setFont("Helvetica", 9)

    classification_report_path = os.path.join(
        model_path, "classification_report_test.txt"
    )
    if os.path.exists(classification_report_path):
        with open(classification_report_path, "r") as f:
            classification_report = f.readlines()
    else:
        classification_report = ["No classification report available"]

    for line in classification_report[:8]:  # Limiting lines to avoid overflow
        c.drawString(50, y, line.strip())
        y -= 12
        if y < 100:
            c.showPage()
            y = 750

    # Add images: Loss & Accuracy Curves
    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 750, "Training Loss & Accuracy Curves")

    img1_path = os.path.join(model_path, "model_training_acc.png")
    img2_path = os.path.join(model_path, "model_training_losses.png")

    if os.path.exists(img1_path):
        c.drawImage(ImageReader(img1_path), 50, 400, width=250, height=200)

    if os.path.exists(img2_path):
        c.drawImage(ImageReader(img2_path), 300, 400, width=250, height=200)

    # Add Confusion Matrix
    img3_path = os.path.join(model_path, "confusion_matrices_train_test.png")

    if os.path.exists(img3_path):
        c.drawImage(ImageReader(img3_path), 50, 100, width=450, height=250)

    c.save()
    print(f"PDF Generated: {pdf_path}")


if __name__ == "__main__":
    latest_model = find_latest_model()
    if latest_model:
        print(f"Using latest model: {latest_model}")
        generate_pdf(latest_model)
    else:
        print("No model found in outputs/")
