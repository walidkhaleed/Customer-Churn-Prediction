# Customer Churn Prediction & ROI Analyzer

## Project Overview
This project focuses on predicting customer churn in the telecom industry and translating those machine learning predictions into direct financial savings. It includes an end-to-end classification pipeline and an interactive web application that acts as a "Churn Risk Analyzer" for stakeholders.

## Core Machine Learning Challenges Handled
* **Class Imbalance:** The dataset featured a severe 73/27 imbalance (Non-Churners vs. Churners). To prevent the model from becoming biased, I utilized **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training data, ensuring the model successfully learned the minority class without testing on synthetic data.
* **Evaluation Metrics:** Discarded standard "Accuracy" in favor of **Precision, Recall, and the Confusion Matrix** to directly measure the model's ability to catch true churners while minimizing wasted retention budgets (False Positives).

## Methodology
* **Feature Engineering:** Utilized One-Hot Encoding and Standard Scaling to map human behavioral text data into a machine-readable matrix.
* **Algorithms Tested:** Logistic Regression (Baseline) vs. Random Forest Classifier. 
* **Business Translation:** Extracted feature importances to prove that contract length, monthly charges, and fiber optic internet instability were the primary drivers of churn. Modeled the financial ROI of the Random Forest predictions, proving significant cost savings compared to doing nothing.

## Files in this Repository
* `Customer_Churn_Prediction.ipynb`: The core Jupyter Notebook containing EDA, SMOTE balancing, model training, evaluation, and simulated ROI calculations.
* `app.py`: The Streamlit web application script.
* `.pkl files`: The compressed Random Forest model, standard scaler, and expected columns array needed to run the app.

## How to Run the App Locally
```bash
git clone [https://github.com/YOUR_USERNAME/Customer-Churn-Prediction.git](https://github.com/YOUR_USERNAME/Customer-Churn-Prediction.git)
cd Customer-Churn-Prediction
pip install -r requirements.txt
streamlit run app.py
