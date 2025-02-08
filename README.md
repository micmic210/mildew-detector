
# Mildew Detection in Cherry Leaves

## Project Overview

This project aims to solve a critical challenge faced by Farmy & Foods, an agricultural company struggling with powdery mildew detection on cherry leaves. Powdery mildew is a fungal disease that negatively impacts crop yield and quality. The current detection process is manual, requiring approximately 30 minutes per tree, making it inefficient and impractical for large-scale farming.

## Project Objectives

To address this issue, this project leverages Machine Learning (ML) and Computer Vision to develop an AI-powered Mildew Detection Dashboard with the following functionalities:
1. Visual Differentiation:
    - Provide data visualizations to help differentiate healthy vs. mildew-infected leaves.
2. Automated Classification:
    - Build a binary classification model that predicts whether a given leaf image is healthy or infected.
3. Scalability:
	- If successful, this system could be scaled and adapted to detect other plant diseases in the future.

By integrating this ML-powered tool into farm operations, Farmy & Foods can significantly reduce labor costs, improve detection accuracy, and increase crop yield.

---

## Dataset Content

The dataset contains images of cherry leaves categorized into two classes:
- **Healthy leaves**
- **Leaves with powdery mildew**

The images were captured from Farmy & Foods’ cherry crops and are publicly available on Kaggle.

- **Dataset Source**: [Cherry Leaves Dataset](https://www.kaggle.com/codeinstitute/cherry-leaves)
- **Dataset Size**: 4208 images

---

## Business Requirements

The client has outlined the following three business requirements:
1.	Visual Differentiation:
	- Conduct a study to visually distinguish healthy cherry leaves from mildew-infected leaves.
2.	Mildew Detection & Classification:
	- Develop a Machine Learning model that predicts whether a given cherry leaf is healthy or infected based on an uploaded image.
3.	Scalability & Future Applications:
	- Analyze the potential to extend the system for detecting other plant diseases and integrate it with IoT and drone-based systems.

---

## Hypotheses and Validation

### Hypotheses
1.	Visual Differences Exist:
    -	Healthy leaves have a uniform texture and consistent color.
	-	Mildew-infected leaves display discoloration, white fungal growth, and surface irregularities.
2.	Machine Learning Can Accurately Detect Mildew:
	-	A well-trained Convolutional Neural Network (CNN) can classify cherry leaves with at least 90% accuracy.
3.	Scalability to Other Crops is Feasible:
	-	The same ML pipeline can be fine-tuned to detect other plant diseases.

### Validation Plan


| **Hypothesis**                          | **Validation Method**                                                                                                                                  | **Success Criteria**                                                                 |
|-----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| **1. Visual Differences**               | Compute **mean & standard deviation images**, analyze **histograms of color distributions**, and create **image montages**.                         | Observable **visual differences** in color, texture, and fungal presence.         |
| **2. ML Classification Feasibility**    | Train and evaluate **CNN and alternative models** (e.g., SVM, Random Forest). Compare **accuracy, precision, recall, and F1-score**.               | CNN achieves **≥ 90% accuracy** with **high recall** for infected leaves.         |
| **3. Scalability to Other Crops**       | Apply **Transfer Learning** on datasets from different crops and test **generalization**.                                                           | Model performs **consistently well across datasets** with **minimal fine-tuning**. |

---

## The Rationale to Map Business Requirements to Data Visualizations and ML Tasks  

| **Business Requirement**             | **Data Visualization & ML Task**                                                                                           |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| **Visual Differentiation**          | - Compute **mean & standard deviation images**.  <br>- Generate **image montages**.  <br>- Compare **histogram distributions** of healthy vs. infected leaves.  |
| **Mildew Detection**                 | - Train a **CNN classifier** with optimized hyperparameters.  <br>- Compare **CNN with Random Forest** baseline.  <br>- Deploy a **Streamlit dashboard** for real-time classification. |
| **Scalability & Future Applications**| - Conduct **Transfer Learning** experiments on other plant diseases.  <br>- Investigate **IoT & drone-based monitoring** feasibility.  <br>- Outline **future model improvements**. |


---

## ML Business Case  

### **Objective**  
To automate **mildew detection** by leveraging **Convolutional Neural Networks (CNNs)** for **binary classification**.

---

### **ML Model Details**  

| **Aspect**            | **Details**                                                        |
|----------------------|------------------------------------------------------------------|
| **ML Task**         | **Binary Classification** (Healthy vs. Infected)                  |
| **Model Architecture** | **CNN** with **data augmentation & dropout**                      |
| **Evaluation Metrics** | **Accuracy, Precision, Recall, F1-score, Confusion Matrix**      |
| **Success Criteria** | **≥ 90% accuracy**, high recall for detecting mildew               |

---

### **ML Pipeline Steps**  

1 **Image Preprocessing**: Resize, Normalize, Augment.  
2 **Model Training**: CNN + **Regularization** (Dropout & L2).  
3 **Evaluation & Hyperparameter Tuning**.  
4 **Deployment via Streamlit & Heroku**.  

---

## Project Workflow: CRISP-DM Framework 

1. **Business Understanding**  
   - **Problem**: **Manual detection is inefficient and unscalable**.  
   - **Goal**: Develop an **ML-based mildew detection system**.  
2. **Data Understanding & Preparation**  
   - Dataset: **4,208 labeled images**.  
   - Processing: **Image cleaning, resizing, augmentation**.  
3. **Modeling**  
   - **Train CNN & Random Forest models**.  
   - **Optimize hyperparameters** for best performance.  
4. **Evaluation**  
   - **Compare model performance** using precision, recall, and accuracy.  
5. **Deployment**  
   - **Deploy an interactive AI-powered Streamlit dashboard**.  

---

##  **Dashboard Design**  

The **interactive Streamlit dashboard** consists of **5 key pages**:

### ** Page 1: Project Overview**  
 **Introduction to Powdery Mildew & its impact**.  
 **Current manual inspection issues & need for AI**.  
 **How this ML-based solution improves efficiency**.  

### ** Page 2: Leaf Visualizer**  
 **Mean & Standard Deviation Images**.  
 **PCA & t-SNE plots for feature separation**.  
 **Chi-Square test & heatmap for key pixel importance**.  

### ** Page 3: Mildew Detector (ML Predictions)**  
 **Upload images & get instant AI classification**.  
 **Confidence scores for each prediction**.  
 **Grad-CAM visual explanations**.  

### ** Page 4: Hypothesis Validation**  
 **Data-driven proof of visual differences**.  
 **Performance metrics: Confusion Matrix, Classification Report**.  
 **Comparing CNN vs. other ML models**.  

### ** Page 5: Model Performance & Future Improvements**  
 **Training history, Accuracy/Loss graphs**.  
 **Evaluation metrics: ROC curve, F1-score**.  
 **Scalability insights & future applications**.  

---

## **User Stories & Acceptance Criteria**  

### ** User Story 1: Visual Differentiation of Healthy & Infected Leaves**  
 **As a** Researcher / Client  
 **I want to** understand **key differences between healthy & infected leaves**.  
 **So that** I can **improve manual detection**.  

####  **Acceptance Criteria**  
- Display **Mean & Standard Deviation images**.  
- Generate **PCA & t-SNE plots** for feature separation.  
- Conduct **Chi-Square test & display heatmaps**.  

---

### ** User Story 2: AI-Powered Disease Prediction**  
 **As a** Farmer / Agricultural Inspector  
 **I want to** use **AI to classify cherry leaves instantly**.  
 **So that** I can **reduce manual labor & improve efficiency**.  

####  **Acceptance Criteria**  
- Train **CNN model (≥99% accuracy)**.  
- Display **Confusion Matrix, Classification Report**.  
- Implement **Grad-CAM for explainability**.  

---

### ** User Story 3: Simple & User-Friendly Web App**  
 **As a** Field Worker / IT Specialist  
 **I want to** use a **web-based tool for real-time mildew detection**.  
 **So that** I can **upload images & get instant results**.  

####  **Acceptance Criteria**  
- Build **Streamlit app with easy image upload**.  
- Display **real-time AI predictions & confidence scores**.  
- Deploy on **Heroku for global access**.  

---
##  Future Work 

 **Expand ML system to detect other plant diseases**.  
 **Integrate IoT & drones for real-time data collection**.  
 **Improve model robustness using transfer learning**. 

 ---

## Unfixed Bugs

- The model’s performance may vary under non-standard conditions (e.g., unusual lighting or damaged leaves)

---

## Deployment

### Heroku


- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- The project was deployed to Heroku following these simplified steps:

1. Log in to Heroku and create an app.
2. Link the app to the GitHub repository containing the project code.
3. Select the branch to deploy and click "Deploy Branch."
4. Once the deployment completes, click "Open App" to access the live app.
5. Ensure that deployment files, such as `Procfile` and `requirements.txt`, are correctly configured.
6. Use a `.slugignore` file to exclude unnecessary large files if the slug size exceeds limits.

### Repository Structure
- **app_pages/**: Streamlit app pages.
- **src/**: Auxiliary scripts (e.g., data preprocessing, model evaluation).
- **notebooks/**: Jupyter notebooks for data analysis and model training.
- **Procfile, requirements.txt, runtime.txt, setup.sh**: Files for Heroku deployment.

---
 ##  Technologies Used  

- **Python** (TensorFlow, Scikit-Learn, OpenCV, Matplotlib, Pandas, NumPy)  
- **Machine Learning**: CNN, SVM, Random Forest  
- **Streamlit** (Dashboard Deployment)  
- **Heroku** (Web App Hosting)  

---

## Credits & Acknowledgements  

- **Farmy & Foods** for dataset contribution.  
- **Kaggle** for hosting the Cherry Leaves dataset.  
- **TensorFlow & Scikit-Learn** documentation for ML techniques.  

---

### Content

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

---
### Media

---


## Acknowledgements
- **Farmy & Foods** for providing the dataset and project inspiration.
- Code Institute for guidance and support in building this project.
- Kaggle for hosting the cherry leaves dataset and enabling access to quality data.
- The contributors of TensorFlow and Scikit-Learn for their excellent documentation and tutorials.
- Community forums and online resources for addressing technical challenges and sharing best practices.

---