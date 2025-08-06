# **Testing**

This document provides a **detailed breakdown of the testing process**, including both **manual testing** and **automated testing with pytest**, to ensure the reliability and accuracy of the mildew detection model and web application.

---

## **Manual Testing**
**Objective:**  
Manual testing was conducted to verify that each **User Story & Business Requirement** is met through functionality checks on the **relevant web pages**.

### Manual Testing Table

| **Test Case ID** | **User Story** | **Test Scenario** | **Test Steps** | **Expected Outcome** | **Web Page** | **Result** |
|-----------------|--------------|---------------|------------|----------------|----------|---------|
| 1 | Visual Differentiation | Display Healthy vs. Infected Leaves | Navigate to "Leaves Visualizer" & check images | Both images are clearly shown | **Leaves Visualizer** | Pass |
| 2 | Visual Differentiation | Mean & Standard Deviation Images | Select checkbox for Avg/Var images | Images load & show visual differences | **Leaves Visualizer** | Pass |
| 3 | Visual Differentiation | t-Test & Heatmap | Run statistical analysis | Displayed test results & heatmap | **Leaves Visualizer** | Pass |
| 4 | AI Prediction | Upload Image for Prediction | Upload an image to detector | AI classifies as Healthy/Infected | **Mildew Detector** | Pass |
| 5 | AI Prediction | View Model Confidence Score | Check histogram | Confidence score displayed | **ML Performance Metrics** | Pass |
| 6 | AI Prediction | Display Classification Report | View test report | Report shows accuracy, precision, recall | **ML Performance Metrics** | Pass |
| 7 | AI Prediction | View Confusion Matrix | Check matrix visualization | Correct classifications shown | **ML Performance Metrics** | Pass |
| 8 | AI Prediction | Display ROC Curve | View ROC curve | AUC score appears, indicating model performance | **ML Performance Metrics** | Pass |
| 9 | AI Prediction | Prediction Probability Analysis | Select test image to analyze confidence score | Confidence score reflects model reliability | **Mildew Detector** | Pass |
| 10 | Web App Usability | Upload multiple images | Upload multiple images at once | All images processed & classified | **Mildew Detector** | Pass |
| 11 | Web App Usability | Generate Prediction Report | Download CSV | CSV file correctly saves predictions | **Mildew Detector** | Pass |
| 12 | Web App Usability | Mobile Responsiveness | Open on mobile/tablet | Layout adjusts correctly | **All Pages** | Pass |
| 13 | Web App Usability | Navigation Across Pages | Click between menu items | Pages load correctly | **All Pages** | Pass |
| 14 | Web App Usability | Page Load Speed | Open web app & test performance | Pages load in under 3 seconds | **All Pages** | Pass |
| 15 | Web App Usability | External Links & Documentation | Click README/Wikipedia links | External links open correctly | **Quick Project Summary** | Pass |

---

## **Automated Testing: pytest**
**Objective:**  
Automated tests were conducted using **pytest** to verify key model and application functionalities.


============================= test session starts ==============================
platform linux -- Python 3.12.1, pytest-8.3.5, pluggy-1.5.0 -- /home/cistudent/.pyenv/versions/3.12.1/bin/python3
cachedir: .pytest_cache
rootdir: /workspaces/mildew-detector
configfile: pytest.ini
plugins: pylama-8.4.1, anyio-4.7.0
collecting ... collected 3 items

tests/test_mildew_detector.py::test_resize_input_image PASSED            [ 33%]
tests/test_mildew_detector.py::test_load_model_and_predict PASSED        [ 66%]
tests/test_mildew_detector.py::test_download_dataframe_as_csv PASSED     [100%]

=============================== warnings summary ===============================
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: Type google._upb._message.MessageMapContainer uses PyType_Spec with a metaclass that has custom tp_new. This is deprecated and will no longer be allowed in Python 3.14.

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: Type google._upb._message.ScalarMapContainer uses PyType_Spec with a metaclass that has custom tp_new. This is deprecated and will no longer be allowed in Python 3.14.

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 3 passed, 2 warnings in 4.98s =========================

- **All tests passed successfully, confirming that core functionalities work as expected.**
- **Two deprecation warnings were noted, which should be monitored for future Python updates.**

## **Automated Testing: PEP 8**
