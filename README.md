# Credit Card Default Prediction – ML Classification Project

## 📌 Problem Statement
The objective of this project is to build and compare multiple machine learning classification models to predict whether a credit card customer will default on payment in the next month.

The project also demonstrates end-to-end ML workflow including:
* Data preprocessing
* Model training
* Model evaluation
* Model comparison
* Streamlit web app deployment

## 📊 Dataset Description
The dataset used is the UCI Credit Card Default Dataset.
* Total Instances: 30,000
* Total Features: 23 input features
* Target Variable: default.payment.next.month
* Type: Binary Classification (0 = No Default, 1 = Default)
### Feature Categories:
* Demographic Information: SEX, EDUCATION, MARRIAGE, AGE
* Credit Limit: LIMIT_BAL
* Payment History: PAY_0 to PAY_6
* Bill Amounts: BILL_AMT1 to BILL_AMT6
* Previous Payments: PAY_AMT1 to PAY_AMT6
* The dataset is imbalanced, with a higher proportion of non-default cases compared to default cases.

## 🤖 Models Implemented

The following six classification models were implemented and evaluated on the same dataset:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

Class imbalance was handled using:
* class_weight='balanced' (for Logistic Regression, Decision Tree, Random Forest)
* scale_pos_weight (for XGBoost)

## 📈 Evaluation Metrics
Each model was evaluated using the following metrics:
* Accuracy
* AUC (Area Under ROC Curve)
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

## 📊 Model Comparison Table
Comparison Table with the evaluation metrics calculated for all the 6
models as below:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC 
| :--- | :---: | :---: | :---: | :---:| :---: | :---: |
| `Logistic Regression` | 0.698167 | 0.725037 | 0.382768 | 0.619193 | 0.473087 | 0.293514
| `Decision Tree` | 0.729000 | 0.609009 | 0.384160 | 0.395278 | 0.389640 | 0.215557
| `KNN` | 0.795000 | 0.707818 | 0.548652 | 0.356436 | 0.432133 | 0.324746
| `Naive Bayes` | 0.381333 | 0.673148 | 0.245815 | 0.883473 | 0.384615 | 0.125253
| `Random Forest (Ensemble)` | 0.815167 | 0.761248 | 0.649560 | 0.337395 | 0.444110 | 0.373071 
| `XGBoost (Ensemble)` | 0.763167 | 0.768018 | 0.467233 | 0.586443 | 0.520095 | 0.369706

Observations on the performance of each model on the chosen
dataset as below:


| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Provides a strong baseline model with balanced performance after handling class imbalance. Performs well in terms of interpretability but limited in capturing complex nonlinear patterns |
| Decision Tree | Able to capture nonlinear relationships but prone to overfitting. Slightly lower generalization compared to ensemble methods |
| kNN | Performance depends heavily on feature scaling. Computationally expensive for larger datasets and moderate predictive performance |
| Naive Bayes | Fast and simple algorithm. However, independence assumption between features limits its predictive capability |
| Random Forest (Ensemble) | Shows strong overall performance due to ensemble averaging. Handles class imbalance effectively and reduces overfitting |
| XGBoost (Ensemble) | Achieved the best AUC and MCC. Effectively captures complex patterns and handles imbalance using scale_pos_weight. Most reliable model for this dataset |

## 🖥 Streamlit Application Features

The deployed Streamlit app includes:

* CSV dataset upload option
* Model selection dropdown
* Display of evaluation metrics
* Confusion matrix visualization
* Classification report

Live App Link:
👉 https://credit-default-app-ml.streamlit.app/

## 📂 Repository Structure

### Project Structure
```text
credit-default-app/
│-- app.py                # Main application script
│-- requirements.txt      # Project dependencies
│-- README.md             # Project documentation
│-- models/               # Saved model binaries
    │-- logistic_model.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- random_forest.pkl
    │-- xgboost.pkl

