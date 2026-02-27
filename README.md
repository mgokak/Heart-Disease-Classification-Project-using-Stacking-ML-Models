# Heart Disease Prediction using Machine Learning Models

## Overview

This project builds a **machine learning model to predict the likelihood of heart disease** using patient clinical data. The system analyzes medical attributes such as age, cholesterol, blood pressure, and other health indicators to classify whether a patient is likely to have heart disease.

The notebook demonstrates a complete **end-to-end machine learning pipeline**, including data exploration, preprocessing, model training, evaluation, and performance analysis.

This project demonstrates:
- Healthcare data analysis
- Exploratory Data Analysis (EDA)
- Feature preprocessing and scaling
- Supervised classification
- Model evaluation and comparison

Applications include early risk screening, clinical decision support, and healthcare analytics.

---

## Project Flow

The system follows a structured machine learning workflow:

1. Import required libraries  
2. Load and explore the dataset  
3. Perform data preprocessing  
4. Split data into training and testing sets  
5. Train machine learning models  
6. Evaluate model performance  
7. Analyze results  

```
Dataset → EDA → Preprocessing → Train/Test Split → Model Training → Evaluation
```

---

## Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Project Structure

```
PredictHeartDisease.ipynb
heart.csv
README.md
```

---

## Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### Explanation

- **pandas / numpy** – Data loading and numerical operations  
- **matplotlib / seaborn** – Data visualization and EDA  
- **scikit-learn** – Model training and evaluation  

---

## Load Dataset

```python
df = pd.read_csv("heart.csv")
df.head()
```

### Explanation

The dataset contains patient health records with features such as:

- Age  
- Sex  
- Chest pain type  
- Resting blood pressure  
- Cholesterol  
- Maximum heart rate  
- Exercise-induced angina  
- Target (presence of heart disease)

---

## Exploratory Data Analysis (EDA)

### Check Dataset Information

```python
df.info()
df.describe()
```

### Check Missing Values

```python
df.isnull().sum()
```

### Correlation Analysis

```python
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()
```

### Explanation

EDA helps understand feature distributions, relationships, and potential predictors of heart disease.

---

## Data Preprocessing

### Separate Features and Target

```python
X = df.drop("target", axis=1)
y = df["target"]
```

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Feature Scaling

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Explanation

- Splits data into training and testing sets  
- Standardization ensures features are on the same scale, improving model performance  

---

## Model Training

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
```

### Explanation

Multiple models are trained to compare performance and identify the best classifier for the problem.

---

## Model Evaluation

```python
y_pred = lr_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Explanation

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics help assess how well the model predicts heart disease risk.

---

## Machine Learning Pipeline (Detailed)

1. Data Loading  
2. Data Exploration  
3. Feature Preparation  
4. Train-Test Split  
5. Feature Scaling  
6. Model Training  
7. Performance Evaluation  

---


## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science
