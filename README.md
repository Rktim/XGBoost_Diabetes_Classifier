# XGBoost Diabetes Classifier

This project implements an **XGBoost** classifier to predict diabetes based on health parameters like glucose levels, BMI, age, and more.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Usage](#usage)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

## Overview
The Diabetes Classifier uses **XGBoost** to predict whether a patient has diabetes or not. The model is trained and evaluated with metrics such as accuracy, confusion matrix, and ROC-AUC score.

## Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd XGBoost_Diabetes_Classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The dataset contains 768 samples with the following features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (target variable)

## Model Training and Evaluation
- Data is split into training and testing sets.
- The **XGBoost classifier** is used for prediction.
- Evaluation metrics include **accuracy**, **confusion matrix**, **ROC AUC score**, and more.

## Usage
Run the script:
```bash
python main.py
```

## Results
- **Accuracy**: 73.96%
- **ROC AUC Score**: 0.74
- **Confusion Matrix**:
  ```
  [[92 31]
   [19 50]]
  ```
  
- **Classification Report**: 
  ```
  Precision: 0.62 (for diabetic cases)
  Recall: 0.72 (for diabetic cases)
  ```
![Screenshot 2024-11-07 230131](https://github.com/user-attachments/assets/4d65f457-72c9-43dc-967c-d4ceb1c16b5e)

## Contributing
Feel free to fork the repository and submit pull requests. Please discuss major changes via issues before making them.

## License
[MIT License](LICENSE)

