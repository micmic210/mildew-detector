import joblib
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate_pdf():
    file_path = "outputs/draft/evaluation.pkl"
    summary_path = "outputs/draft/model_summary.txt"
    pdf_path = "outputs/draft/trial_report.pdf"

    # Check if evaluation file exists
    if not os.path.exists(file_path):
        print("No evaluation file found. Run the notebook first.")
        return

    # Load evaluation results
    results = joblib.load(file_path)

    # Load model summary
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            model_summary = f.read()
    else:
        model_summary = "No model summary available"

    # Create PDF
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 750, "Model Evaluation Report")

    c.setFont("Helvetica", 11)
    y = 720

    # Print Model Information
    model_info = {
        "Batch Size": results.get("batch_size", "N/A"),
        "Image Shape": str(results.get("image_shape", "N/A")),
        "Optimizer": results.get("optimizer", "N/A"),
        "Loss Function": results.get("loss_function", "N/A"),
        "Total Parameters": results.get("total_params", "N/A"),
        "Trainable Parameters": results.get("trainable_params", "N/A"),
        "Non-Trainable Parameters": results.get("non_trainable_params", "N/A"),
    }

    for key, value in model_info.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    y -= 10  # Add some spacing

    # Add Model Summary
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

    # Print Evaluation Metrics
    evaluation_metrics = {
        "Accuracy": round(results.get("accuracy", 0), 3),
        "Validation Accuracy": round(results.get("val_accuracy", 0), 3),
        "Loss": round(results.get("loss", 0), 3),
        "Validation Loss": round(results.get("val_loss", 0), 3),
    }

    for key, value in evaluation_metrics.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    # Save and Close PDF
    c.save()
    print(f"PDF Generated: {pdf_path}")


if __name__ == "__main__":
    generate_pdf()