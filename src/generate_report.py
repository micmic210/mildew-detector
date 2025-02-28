import joblib
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate_pdf():
    file_path = "outputs/draft/evaluation.pkl"

    if not os.path.exists(file_path):
        print("No evaluation file found. Run the notebook first.")
        return

    results = joblib.load(file_path)
    pdf_path = "outputs/draft/trial_report.pdf"

    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, "Model Evaluation Report")

    c.setFont("Helvetica", 11)
    y = 720

    # Print Model Information (with defaults)
    c.drawString(50, y, f"Batch Size: {results.get('batch_size', 'N/A')}")
    y -= 20
    c.drawString(50, y, f"Image Shape: {str(results.get('image_shape', 'N/A'))}")
    y -= 20
    c.drawString(50, y, f"Optimizer: {results.get('optimizer', 'N/A')}")
    y -= 20
    c.drawString(50, y, f"Loss Function: {results.get('loss_function', 'N/A')}")
    y -= 20
    c.drawString(50, y, f"Total Parameters: {results.get('total_params', 'N/A')}")
    y -= 20
    c.drawString(
        50, y, f"Trainable Parameters: {results.get('trainable_params', 'N/A')}"
    )
    y -= 20
    c.drawString(
        50, y, f"Non-Trainable Parameters: {results.get('non_trainable_params', 'N/A')}"
    )
    y -= 30

    # Add Model Summary (if available)
    model_summary = results.get("model_summary", "No model summary available")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Model Summary:")
    y -= 20
    c.setFont("Helvetica", 10)

    for line in model_summary.split("\n"):
        c.drawString(50, y, line)
        y -= 15
        if y < 100:
            c.showPage()
            y = 750

    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, "Evaluation Results")

    c.setFont("Helvetica", 11)
    y = 720
    c.drawString(50, y, f"Accuracy: {round(results.get('accuracy', 0), 3)}")
    y -= 20
    c.drawString(
        50, y, f"Validation Accuracy: {round(results.get('val_accuracy', 0), 3)}"
    )
    y -= 20
    c.drawString(50, y, f"Loss: {round(results.get('loss', 0), 3)}")
    y -= 20
    c.drawString(50, y, f"Validation Loss: {round(results.get('val_loss', 0), 3)}")
    y -= 30

    # Save and Close PDF
    c.save()
    print(f"PDF Generated: {pdf_path}")


if __name__ == "__main__":
    generate_pdf()
