# Heart Disease Classification using Advanced Machine Learning (Stacking & OOF)

## Overview

This project builds an advanced **machine learning system to predict heart disease** using patient clinical data. The system analyzes medical attributes such as age, cholesterol, blood pressure, and other health indicators to classify whether a patient is likely to have heart disease.

The notebook demonstrates a complete **end-to-end machine learning pipeline**, including data exploration, preprocessing, model training, evaluation, and performance analysis.

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature scaling
- Multiple base machine learning models
- Out-of-Fold (OOF) predictions
- Stacking ensemble model
- Performance evaluation and comparison

The objective is to improve predictive performance by combining multiple models into a robust ensemble, similar to industry and Kaggle-style workflows.

Applications include:
- Early heart disease risk screening
- Clinical decision support
- Healthcare analytics and research

---

## Project Flow

The system follows an advanced pipeline:

1. Load and explore dataset  
2. Data preprocessing and scaling  
3. Train multiple base models  
4. Generate Out-of-Fold (OOF) predictions  
5. Use OOF outputs as meta-features  
6. Train stacking (meta) model  
7. Evaluate final ensemble performance  

```
Data → Preprocessing → Base Models → OOF Predictions → Stacking → Final Prediction
```

---

## Installation

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

from sklearn.model_selection import train_test_split, KFold
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

The dataset contains patient health records with features such as:

- Age  
- Sex  
- Chest pain type  
- Resting blood pressure  
- Cholesterol  
- Maximum heart rate  
- Exercise-induced angina  
- Target variable (heart disease: 0/1)

---


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

## Model Architecture

**Level‑1 (Base Models)**  
- Logistic Regression  
- KNN  
- SVM  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- AdaBoost  

**Level‑2 (Meta Model)**  
- Logistic Regression (Stacking)

---

## Base Models Used (Level‑1)

The following models are trained as base learners:

1. Logistic Regression  
2. K-Nearest Neighbors (KNN)  
3. Support Vector Machine (SVM)  
4. Decision Tree  
5. Random Forest  
6. Gradient Boosting  
7. AdaBoost  

Example:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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


```

Each model captures different patterns and decision boundaries.

---

## Out-of-Fold (OOF) Predictions

OOF predictions are generated using K-Fold cross-validation.

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((X_train.shape[0], n_models))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model.fit(X_tr, y_tr)
    oof_preds[val_idx, model_idx] = model.predict_proba(X_val)[:, 1]
```

### Why OOF?

- Prevents data leakage  
- Provides unbiased validation predictions  
- Creates reliable meta-features for stacking  

---

## Stacking Model (Level‑2)

OOF predictions from base models are combined to train a meta-model.

```python
from sklearn.linear_model import LogisticRegression

meta_model = LogisticRegression()
meta_model.fit(oof_preds, y_train)
```

Test-time stacking:

```python
meta_test = np.column_stack([
    model.predict_proba(X_test)[:, 1] for model in base_models
])

final_preds = meta_model.predict(meta_test)
```

---

## Model Evaluation

```python
print("Accuracy:", accuracy_score(y_test, final_preds))
print(confusion_matrix(y_test, final_preds))
print(classification_report(y_test, final_preds))
```

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---


## Practical Applications

- Early heart disease risk detection  
- Clinical decision support systems  
- Patient risk stratification  
- Healthcare analytics  

---

## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science
