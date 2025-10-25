# ML-Based-Predictive-Modeling-for-Early-Detection-of-Subclinical-Mastitis-in-Dairy-Herds
🐄 Subclinical Mastitis Detection using Machine Learning
📘 Project Overview

This project aims to detect subclinical mastitis in dairy cows using machine learning techniques and simple, cost-effective milk quality indicators such as:

Somatic Cell Count (SCC)

Electrical Conductivity

Milk pH

Fat %, Protein %, and Lactose %

The model identifies early signs of mastitis before clinical symptoms appear, enabling timely treatment and improving milk productivity.

🎯 Objectives

Develop a predictive ML model for mastitis detection.

Reduce dependency on expensive biomarkers (like LDH).

Improve dairy herd health management through data-driven decisions.

⚙️ Features

LightGBM-based predictive model trained on multiple milk parameters.

Data preprocessing pipeline with feature scaling and transformation.

Easily deployable in an IoT-based dairy monitoring system.

Supports future integration with mobile or web dashboards.

🧠 Machine Learning Models Used

LightGBM (final model) — best accuracy and AUC.

XGBoost and Random Forest — evaluated for comparison.

Feature engineering includes:

Log transformation of SCC

Derived ratios (SCC/fat%)

Normalization & scaling

🧩 Project Structure
📦 Mastitis-Detection
 ┣ 📜 sbmastitis_fixed.ipynb       ← Cleaned Colab notebook
 ┣ 📜 scaler.joblib                ← Feature scaler
 ┣ 📜 lgbm_mastitis_model.joblib   ← Trained LightGBM model
 ┣ 📜 README.md                    ← Project documentation
 ┣ 📜 requirements.txt             ← Python dependencies
 ┗ 📂 data/                        ← Sample dataset

🧪 Example Prediction
import joblib
import numpy as np

# Load model and scaler
lgbm = joblib.load("lgbm_mastitis_model.joblib")
scaler = joblib.load("scaler.joblib")

# Example new sample (SCC, pH, conductivity, logSCC, SCC/fat, etc.)
sample = [[250000, 6.8, 5.3, np.log1p(250000), 250000/5.3, 250000, 22.0, 3.7, 3.2]]
sample_scaled = scaler.transform(sample)

# Predict
prediction = lgbm.predict(sample_scaled)
print("Mastitis detected" if prediction[0] == 1 else "Healthy cow")

📊 Results Summary
Metric	LightGBM	XGBoost	Random Forest
Accuracy	96.2%	94.7%	93.5%
Precision	95.8%	93.1%	91.6%
Recall	96.5%	94.2%	92.4%
🧩 Requirements

Create a requirements.txt with:

numpy
pandas
scikit-learn
lightgbm
joblib
matplotlib


Install with:

pip install -r requirements.txt
