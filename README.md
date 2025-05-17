# 💧 Predicting Molecular Solubility (LogS) using Machine Learning

This project aims to predict the solubility (LogS) of chemical compounds using molecular descriptors. It is my first machine learning project built using Python in Google Colab.

---

## 📊 Dataset

- Source: [Data Professor GitHub](https://github.com/dataprofessor/data)
- File: `delaney_solubility_with_descriptors.csv`
- Target: `logS` (Log of solubility)
- Features: Molecular descriptors (e.g., MolWt, MolLogP, NumRotatableBonds, etc.)

---

## 🧪 Project Steps

1. **Load the dataset** from a public URL.
2. **Prepare the data** by separating features (X) and the target variable (y).
3. **Split the data** into training and testing sets.
4. **Train models** using:
   - Linear Regression
   - Random Forest Regressor
5. **Evaluate models** using Mean Squared Error (MSE) and R-squared (R²).
6. **Compare performance** of both models.
7. **Visualize predictions** to assess accuracy.

---

## 📈 Results Overview

| Model            | Train MSE | Train R² | Test MSE | Test R² |
|------------------|-----------|----------|----------|---------|
| Linear Regression|    ...    |   ...    |   ...    |   ...   |
| Random Forest    |    ...    |   ...    |   ...    |   ...   |

> Replace `...` with actual results from the notebook.

---

## 🔧 Tools & Libraries

- Python
- Pandas
- Scikit-learn
- NumPy
- Matplotlib
- Google Colab

---

## 🚀 Future Improvements

- Add more models (e.g., XGBoost)
- Hyperparameter tuning
- Save trained models
- Use feature selection or dimensionality reduction

---

## 🙌 Acknowledgments

- Dataset by [Data Professor](https://github.com/dataprofessor/)
- Project inspired by basic ML workflows in chemistry

