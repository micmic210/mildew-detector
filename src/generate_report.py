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

    # Print Model Information
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

    # Add Model Summary
    model_summary_path = "outputs/draft/model_summary.txt"
    if os.path.exists(model_summary_path):
        with open(model_summary_path, "r") as f:
            model_summary = f.readlines()
    else:
        model_summary = ["No model summary available"]

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Model Summary:")
    y -= 20
    c.setFont("Helvetica", 10)

    for line in model_summary:
        c.drawString(50, y, line.strip())
        y -= 15
        if y < 100:
            c.showPage()
            y = 750

    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, "Evaluation Results")

    c.setFont("Helvetica", 11)
    y = 720
    c.drawString(50, y, f"Accuracy: {round(results.get('test_accuracy', 0), 3)}")
    y -= 20
    c.drawString(
        50, y, f"Validation Accuracy: {round(results.get('val_accuracy', 0), 3)}"
    )
    y -= 20
    c.drawString(50, y, f"Loss: {round(results.get('test_loss', 0), 3)}")
    y -= 20
    c.drawString(50, y, f"Validation Loss: {round(results.get('val_loss', 0), 3)}")
    y -= 30

    # Add Confusion Matrix
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Confusion Matrix:")
    y -= 20
    c.setFont("Helvetica", 10)

    confusion_matrix = results.get("confusion_matrix", [[0, 0], [0, 0]])
    c.drawString(50, y, f" {confusion_matrix[0]}")
    y -= 15
    c.drawString(50, y, f" {confusion_matrix[1]}")
    y -= 30

    # Add Classification Report
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Classification Report:")
    y -= 20
    c.setFont("Helvetica", 10)

    classification_report = results.get("classification_report", {})
    for label, metrics in classification_report.items():
        if isinstance(metrics, dict):
            c.drawString(50, y, f"{label}:")
            y -= 15
            for metric, value in metrics.items():
                c.drawString(70, y, f"{metric}: {round(value, 3)}")
                y -= 15
        else:
            c.drawString(50, y, f"{label}: {round(metrics, 3)}")
            y -= 20
        if y < 100:
            c.showPage()
            y = 750

    # Save and Close PDF
    c.save()
    print(f"PDF Generated: {pdf_path}")


if __name__ == "__main__":
    generate_pdf()
