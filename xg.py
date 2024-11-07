from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report, roc_auc_score, roc_curve, auc,f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/raktm/rk/xgboost/diabetes.csv")

#DATA INFO
print("Head :",df.head())
print("Tail :",df.tail())
print("Info :",df.info())
print("Describe :",df.describe())
print("Shape :",df.shape)

print("null count :",df.isnull().sum())

with open("data_info.txt", "w") as f:
    f.write("Head :\n{}\n".format(df.head()))
    f.write("Tail :\n{}\n".format(df.tail()))
    f.write("Info :\n{}\n".format(df.info()))
    f.write("Describe :\n{}\n".format(df.describe()))
    f.write("null count :\n{}\n".format(df.isnull().sum()))

#DATA PREPROCESSING
X_data=df.drop("Outcome",axis=1)
y_data=df["Outcome"]
print("X_data :",X_data)
print("\n y_data :",y_data)

#SPLITTING DATA
xtrain,xtest,ytrain,ytest=train_test_split(X_data,y_data,test_size=0.25,random_state=42)

print("Xtrain :",xtrain)
print("Xtest :",xtest)
print("ytrain :",ytrain)
print("ytest :",ytest)

#MODEL TRAINING
model=XGBClassifier()
model.fit(xtrain,ytrain)

Ypred=model.predict(xtest)
print("Ypred :",Ypred)

#PERFORMANCE EVALUATION
print("Accuracy :",accuracy_score(ytest,Ypred))
print("Confusion Matrix :",confusion_matrix(ytest,Ypred))
print("Classification Report :",classification_report(ytest,Ypred))
print("ROC AUC Score :",roc_auc_score(ytest,Ypred))
print("ROC Curve :",roc_curve(ytest,Ypred))

with open("performance.txt", "w") as f:
    f.write("Accuracy: {}\n".format(accuracy_score(ytest,Ypred)))
    f.write("Confusion Matrix:\n{}\n".format(confusion_matrix(ytest,Ypred)))
    f.write("Classification Report:\n{}\n".format(classification_report(ytest,Ypred)))
    f.write("ROC AUC Score: {}\n".format(roc_auc_score(ytest,Ypred)))
    f.write("ROC Curve:\n{}\n".format(roc_curve(ytest,Ypred)))

#PLOTTING
correct = (ytest == Ypred)
incorrect = ~correct

plt.figure(figsize=(14, 8))

plt.scatter(np.arange(1, len(ytest) + 1)[correct], ytest[correct], 
            label="Correct Prediction", color='green', s=100, alpha=0.6, marker='o')

plt.scatter(np.arange(1, len(ytest) + 1)[incorrect], ytest[incorrect], 
            label="Incorrect Prediction", color='red', s=100, alpha=0.6, marker='x')

plt.scatter(np.arange(1, len(Ypred) + 1), Ypred, label="Model Prediction", color='orange', s=80, alpha=0.6)

plt.axhline(y=0.5, color='purple', linestyle='--', label='Decision Boundary (0.5)')

plt.title("Diabetes Prediction: Actual vs Model Predictions", fontsize=18)
plt.xlabel("Sample Index", fontsize=14)
plt.ylabel("Outcome (1 = Diabetic, 0 = Non-Diabetic)", fontsize=14)
plt.yticks([0, 1], labels=["Non-Diabetic", "Diabetic"], fontsize=12)

plt.legend(loc="upper right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
