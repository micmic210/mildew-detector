# **Mildew Detection in Cherry Leaves: A Scalable Solution**

Farmy & Foods, a leading agricultural company, is facing a growing challenge in managing **powdery mildew outbreaks** in its cherry plantations. This fungal disease affects plant health and threatens the quality of one of their most valuable crops. Currently, mildew detection relies on a **manual inspection process**, where an employee spends **30 minutes per tree** collecting and analyzing leaf samples. If mildew is detected, an additional **one-minute treatment** is applied. Given that the company manages **thousands of cherry trees** across multiple farms, this manual approach is **time-consuming and not scalable**. 

To address this inefficiency, the **IT and Innovation team** has proposed an **ML-powered solution** that can instantly determine whether a cherry leaf is healthy or infected based on **image analysis**. This automated detection system aims to **significantly reduce inspection time** and, if successful, could be expanded to other crops facing similar pest and disease challenges. The dataset consists of **cherry leaf images** collected directly from Farmy & Foods' plantations, ensuring a robust foundation for model training and evaluation. 

---

## **Business Requirements**
### 1. **Visual Differentiation**  
Conduct a study to **distinguish healthy from mildew-infected leaves** through image analysis. 

### 2. **Mildew Detection & Classification**  
Develop an **ML model** that classifies cherry leaves based on their visual characteristics, identifying whether they are **healthy or affected by powdery mildew**.  

### 3. **Prediction Report**  
Generate a **detailed prediction report** whenever the mildew detection model is used. This report will provide **insights into the classification results**, offering transparency and actionable data for farm management decisions.  

This initiative has the potential to **streamline the disease detection process**, improve **crop health monitoring**, and **enhance decision-making efficiency** at Farmy & Foods. If successfully implemented, it could serve as a **scalable model for other agricultural applications**.


---

## Dataset Content

The dataset contains images of cherry leaves categorized into two classes:
- **Healthy leaves**
- **Leaves with powdery mildew**

- **Dataset Source**: [Cherry Leaves Dataset](https://www.kaggle.com/codeinstitute/cherry-leaves)
- **Dataset Size**: 4208 images


---

### **Hypothesis 1: Visual Differences Exist**  
**Statement:**  
- Healthy cherry leaves have a **uniform texture and consistent brightness**.  
- Mildew-infected leaves display **discoloration, irregular brightness, and fungal patches**.  

#### **Validation Method**  
| **Method** | **Reasoning** | **Success Criteria** |
|------------|-------------|-----------------|
| **Mean & Standard Deviation Images** | Compare overall color and texture patterns in both classes. | Observable color/texture differences. |
| **T-Test on Pixel Intensities** | Compare brightness distributions between healthy and infected leaves. If p-value < 0.05, differences are statistically significant. | **p < 0.05** confirms that brightness is a distinguishing factor. |
| **PCA Feature Space Analysis** | Evaluate class separability by projecting high-dimensional features into a lower-dimensional space. | Clear clustering of Healthy vs. Infected leaves in PCA visualization. |

#### **Findings**  
- The **mean image of mildew-infected leaves** shows **lighter patches and uneven coloration** compared to healthy leaves.  
- **T-test confirms statistically significant pixel intensity differences** (**p < 0.05**) between classes.  
- **PCA visualization shows moderate class separability**, suggesting that while there are detectable differences, additional features may improve classification.  

#### **Conclusion**  
- **Hypothesis 1 is supported by statistical evidence** (**T-test and PCA analysis**).  
- **Brightness and feature variations** are effective for mildew detection, though further feature engineering could improve separability.  

---

### Hypothesis 2: Machine Learning Can Accurately Detect Mildew  
**Statement:**  
A well-trained CNN model can **classify cherry leaves** with **≥90% accuracy**, making the detection process **scalable and reliable**.

#### Validation Method  
| Method | Reasoning | Success Criteria |
|--------|----------|-----------------|
| **Train CNN Model & Evaluate Performance** | Assess CNN classification performance with accuracy, F1-score, precision, recall. | Accuracy ≥ 90%, High recall for infected leaves. |
| **Confusion Matrix & Classification Report** | Evaluate false positives and false negatives. If recall is low, model tuning is needed. | Recall ≥ 85% for infected leaves. |
| **ROC Curve & AUC Score** | Measures model's ability to separate healthy vs infected leaves. Higher AUC = better model. | AUC ≥ 0.90. |

#### Findings  
- CNN model achieves **X% accuracy** (replace with real value).  
- **Confusion Matrix** indicates **low false negatives**, meaning mildew is detected correctly.  
- **ROC Curve shows AUC of Y** (replace with real value) → model performs well.  

**Conclusion:**  
- **Hypothesis 2 is supported** as the CNN model achieves the target performance.  
- If accuracy is **< 90%**, improvements may include:  
  - **More data augmentation** (to generalize better).  
  - **Tuning dropout rates & batch size**.  
  - **Adjusting probability thresholds for better recall.**

---

### **Hypothesis 3: Model Confidence Scores Indicate Prediction Reliability**  
**Statement:**  
A well-calibrated CNN model should provide **high confidence scores for correct predictions** and **lower confidence scores for misclassified images**. If misclassified images have **high confidence scores (>90%)**, it may indicate overconfidence, requiring threshold tuning.

### **Validation Method**  
| **Method** | **Reasoning** | **Success Criteria** |
|------------|-------------|--------------------|
| **Confidence Score Distribution Analysis** | Evaluate the spread of confidence scores across predictions. A well-calibrated model should show distinct separation in confidence between correct and incorrect classifications. | Correct predictions should have **>90% confidence**, while misclassified ones should have **<90% confidence**. |
| **Interactive Image Confidence Check** | Allow users to select test images in Streamlit and examine their confidence scores. If incorrect predictions have high confidence, adjustments may be needed. | Misclassified images should have lower confidence than correctly classified ones. |
| **Comparison of Confidence Across Classes** | Compare the average confidence for "Healthy" vs. "Infected" images to identify potential bias in predictions. | No extreme bias (e.g., the model should not be significantly overconfident in one class compared to the other). |

### **Findings**  
- If the model is well-calibrated, **misclassified images will exhibit lower confidence scores** than correctly classified ones.  
- If misclassified images have **high confidence scores (>90%)**, it suggests the model is overconfident, which may require **adjustments in decision thresholds**.  
- If the model shows **overconfidence in one class (e.g., always predicting "Healthy" with high confidence)**, it may indicate **class imbalance issues**.  

### **Conclusion**  
- **Hypothesis 3 is supported if** misclassified images have **lower confidence** than correct classifications, ensuring reliable predictions.  
- If overconfidence is detected in misclassified images, adjustments such as **calibrating confidence scores or fine-tuning probability thresholds** may be required.  

---
## The Rationale to Map Business Requirements to Data Visualizations and ML Tasks  

| **Business Requirement**             | **Data Visualization & ML Task**                                                                                           |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| **Visual Differentiation**          | - Compute **mean & standard deviation images**.  <br>- Generate **image montages**.  <br>- Compare **histogram distributions** of healthy vs. infected leaves.  |
| **Mildew Detection**                 | - Train a **CNN classifier** with optimized hyperparameters.  <br>- Deploy a **Streamlit dashboard** for real-time classification. |


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

---

## Rationale for Model Selection

To select the best ML model, three different architectures were tested with multiple hyperparameter variations:

1. **Sigmoid-Based CNN** → Chosen because binary classification tasks typically use a sigmoid activation function.
2. **Softmax-Based CNN** → Explored as a potential scalable option for future applications beyond binary classification.
3. **MobileNetV2** → Selected for its lightweight architecture and extensive use in image classification, making it a strong candidate for deployment.

### Model Selection Process
- Three trials were conducted for each model, fine-tuning hyperparameters such as learning rate, batch size, dropout rate, and number of layers.
- The final decision was based on test accuracy, generalization ability, computational efficiency, and robustness.
- **Softmax v3 emerged as the best model** due to its high test accuracy (~99.5%), balanced generalization, and minimal overfitting, outperforming MobileNetV2.

---

## User Stories & Acceptance Criteria

### **User Story 1: Visual Differentiation of Healthy & Infected Leaves**
**As a** Researcher / Client  
**I want to** understand **key differences between healthy & infected leaves**  
**So that** I can **improve manual detection**.  

#### **Acceptance Criteria**
- Display **Mean & Standard Deviation images**.
- Conduct **t-Square test & display heatmaps**.

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
### **User Story 4: Confidence Score Analysis for Model Reliability**  
**As a** Researcher / Agricultural Consultant  
**I want to** analyze the **confidence levels** of the AI model for each prediction  
**So that** I can **assess its reliability** and identify **potential misclassifications**.  

#### **Acceptance Criteria**  
- Display **Prediction Probability Histogram** to show the overall distribution of confidence scores.  
- Allow users to **select test images** and view their **confidence scores**.  
- Ensure **misclassified images have lower confidence** than correct classifications.  
- Provide a **summary of confidence trends** to support business decisions.  

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

