# XGBoost Diabetes Classifier

🚀 **Predict Diabetes Risk with XGBoost!**

This project builds a powerful diabetes classifier using the **XGBoost** algorithm, trained on the **PIMA Indians Diabetes Dataset**. The goal is to predict whether a patient is at **High Risk** or **Low Risk** of having diabetes based on various health metrics.

---

## 📌 Features
- **Data Preprocessing:** Handles missing values, scales features with MinMaxScaler, and encodes categorical data.
- **Model Training:** Implements XGBoost with hyperparameter tuning for maximum performance.
- **Evaluation Metrics:** Includes Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Explainability:** Uses SHAP values to provide interpretable insights.
- **Visualizations:** Generates a confusion matrix, ROC curve, and SHAP feature importance plot.
- **Command-line Friendly:** Easily train and test with CLI commands.

---

## 🔧 Installation
```bash
# Clone the repository
git clone https://github.com/Rktim/XGBoost_Diabetes_Classifier.git

# Navigate to project folder
cd XGBoost_Diabetes_Classifier

# Install required packages
pip install -r requirements.txt
```

---

## 📊 Model Performance

### 🎯 Confusion Matrix
The confusion matrix shows how well the model distinguishes between **Low Risk** and **High Risk** patients.

![image](https://github.com/user-attachments/assets/f52b31d6-77b9-4a4c-bd98-0e036adf008d)


- **81** True Negatives (Low Risk correctly predicted)
- **48** True Positives (High Risk correctly predicted)
- **18** False Positives (incorrectly predicted High Risk)
- **7** False Negatives (missed High Risk cases)

📌 **Accuracy:** 86.5%  
📌 **Precision (High Risk):** 72.7%  
📌 **Recall (High Risk):** 87.3%  
📌 **F1-Score:** 79.3%  

---

### 📈 ROC Curve
The ROC curve illustrates the model's ability to distinguish between the two classes.

![image](https://github.com/user-attachments/assets/44dcd685-906c-4f95-88ba-002d464f7084)


- **AUC Score:** 0.85 — A high AUC indicates the model performs well in classification.

---

### 🔍 SHAP Feature Importance
SHAP values explain how each feature contributes to predictions.

![image](https://github.com/user-attachments/assets/1e5ce7fb-3ca8-4bf4-a235-b330bbff5f95)


Key insights:
- **Glucose** is the most significant feature — higher glucose levels push predictions toward "High Risk."
- **BMI** and **Age** are also strong indicators.
- **Blood Pressure** and **Insulin** have mixed contributions, indicating more nuanced influence.

---



## 🔥 Future Improvements
- 📌 **Hyperparameter optimization with Optuna** for faster tuning
- 📌 **Incorporate more diverse clinical datasets** to improve generalizability
- 📌 **Deploy using Streamlit** for an interactive web app
- 📌 **Add cross-validation pipelines** for more robust performance estimates

---

## 🏅 Credits
- **Author:** Rktim
- **Dataset:** PIMA Indians Diabetes Dataset
- **Libraries:** XGBoost, SHAP, Matplotlib, Pandas, NumPy, Scikit-learn

⭐ If you found this project helpful, please **star** the repo!

Happy predicting! 🚀

