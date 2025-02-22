# Mildew Detection in Cherry Leaves

## Project Overview

This project aims to solve a critical challenge faced by Farmy & Foods, an agricultural company struggling with powdery mildew detection on cherry leaves. Powdery mildew is a fungal disease that negatively impacts crop yield and quality. The current detection process is manual, requiring approximately 30 minutes per tree, making it inefficient and impractical for large-scale farming.

## Project Objectives

To address this issue, this project leverages Machine Learning (ML) and Computer Vision to develop an AI-powered Mildew Detection Dashboard with the following functionalities:

1. **Visual Differentiation:**
    - Provide data visualizations to help differentiate healthy vs. mildew-infected leaves.
2. **Automated Classification:**
    - Build a binary classification model that predicts whether a given leaf image is healthy or infected.

By integrating this ML-powered tool into farm operations, Farmy & Foods can significantly reduce labor costs, improve detection accuracy, and increase crop yield.

---

## Dataset Content

The dataset contains images of cherry leaves categorized into two classes:
- **Healthy leaves**
- **Leaves with powdery mildew**

- **Dataset Source**: [Cherry Leaves Dataset](https://www.kaggle.com/codeinstitute/cherry-leaves)
- **Dataset Size**: 4208 images

---

## Business Requirements

1. **Visual Differentiation:**
    - Conduct a study to distinguish healthy from mildew-infected leaves.
2. **Mildew Detection & Classification:**
    - Develop an ML model that classifies cherry leaves based on image analysis.

---

## Hypotheses and Validation

### **Hypotheses**
1. **Visual Differences Exist:**
    - Healthy leaves have a uniform texture and consistent color.
    - Mildew-infected leaves display discoloration, white fungal growth, and surface irregularities.
2. **Machine Learning Can Accurately Detect Mildew:**
    - A well-trained Convolutional Neural Network (CNN) can classify cherry leaves with at least 90% accuracy.

### **Validation Plan**

| **Hypothesis**                  | **Validation Method**                                                                 | **Success Criteria**                            |
|---------------------------------|--------------------------------------------------------------------------------------|------------------------------------------------|
| **1. Visual Differences**       | Compute **mean & standard deviation images**, analyze **histograms of color distributions**. | Observable **visual differences** in color and texture. |
| **2. ML Classification Feasibility** | Train and evaluate **CNN model**. Compare **accuracy, precision, recall, and F1-score**. | CNN achieves **≥ 90% accuracy** with **high recall** for infected leaves. |

---

## Data Processing Pipeline

### **Data Collection**
- **Objectives:**
  - Authenticate and retrieve dataset from Kaggle.
  - Organize data into train (70%), validation (10%), and test (20%) splits.
  - Remove non-image files to ensure data integrity.
- **Outputs:** Cleaned, structured dataset ready for modeling.

### **Data Visualization**
- **Objectives:**
  - Provide insights into dataset composition.
  - Generate image montages and PCA plots.
  - Create pixel intensity histograms for class differentiation.
- **Outputs:**
  - Class distribution analysis.
  - Mean & variability image visualizations.
  - PCA & t-SNE plots for feature separability.

### **Modeling & Evaluation**
- **Objectives:**
  - Train a baseline CNN model.
  - Apply L2 regularization and Dropout.
  - Optimize hyperparameters using Keras Tuner.
  - Evaluate model performance using classification metrics.
  - Implement **Saliency Map** for model explainability.
- **Outputs:**
  - Best-performing model selected based on cross-validation.
  - Confusion Matrix, Learning Curve, and Classification Report.
  - **Saliency Map** integrated for visual interpretability.

---

## The Rationale to Map Business Requirements to Data Visualizations and ML Tasks  

| **Business Requirement**             | **Data Visualization & ML Task**                                                                                           |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| **Visual Differentiation**          | - Compute **mean & standard deviation images**.  <br>- Generate **image montages**.  <br>- Compare **histogram distributions** of healthy vs. infected leaves.  |
| **Mildew Detection**                 | - Train a **CNN classifier** with optimized hyperparameters.  <br>- Deploy a **Streamlit dashboard** for real-time classification. |

---

## User Stories & Acceptance Criteria

### **User Story 1: Visual Differentiation of Healthy & Infected Leaves**
**As a** Researcher / Client  
**I want to** understand **key differences between healthy & infected leaves**  
**So that** I can **improve manual detection**.  

#### **Acceptance Criteria**
- Display **Mean & Standard Deviation images**.
- Generate **PCA & t-SNE plots** for feature separation.
- Conduct **Chi-Square test & display heatmaps**.

---

### **User Story 2: AI-Powered Disease Prediction**
**As a** Farmer / Agricultural Inspector  
**I want to** use **AI to classify cherry leaves instantly**  
**So that** I can **reduce manual labor & improve efficiency**.  

#### **Acceptance Criteria**
- Train **CNN model (≥90% accuracy)**.
- Display **Confusion Matrix, Classification Report**.
- Implement **Saliency Map for explainability**.

---

### **User Story 3: Simple & User-Friendly Web App**
**As a** Field Worker / IT Specialist  
**I want to** use a **web-based tool for real-time mildew detection**  
**So that** I can **upload images & get instant results**.  

#### **Acceptance Criteria**
- Build **Streamlit app with easy image upload**.
- Display **real-time AI predictions & confidence scores**.
- Deploy on **Heroku for global access**.

---

## **Deployment**

### **Heroku Deployment**
- **Live App:** `https://.herokuapp.com/`
- Steps:
  1. Create a Heroku app.
  2. Link GitHub repo & deploy branch.
  3. Configure **Procfile** & **requirements.txt**.
  4. Deploy & test the live application.

---

## **Acknowledgements**

- **Farmy & Foods** for dataset contribution.  
- **Kaggle** for hosting the Cherry Leaves dataset.  
- **TensorFlow & Scikit-Learn** for ML documentation & support.  
- **Code Institute** for guidance in project structuring.  
- **Online ML communities & forums** for troubleshooting support.  

---

