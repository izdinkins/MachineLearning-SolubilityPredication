# 💧 Predicting Molecular Solubility (LogS) using Machine Learning

This is my **first machine learning project**, built in Google Colab using Python. The objective is to **predict the solubility (LogS) of chemical compounds** based on molecular descriptors using machine learning techniques like **Linear Regression** and **Random Forest**.

---

## 📁 Dataset

We use a publicly available dataset from [Data Professor](https://github.com/dataprofessor/data):

- Dataset: [`delaney_solubility_with_descriptors.csv`](https://github.com/dataprofessor/data/blob/master/delaney_solubility_with_descriptors.csv)
- Target variable: `logS` (Log of solubility)
- Features: Molecular descriptors such as `MolLogP`, `MolWt`, `NumRotatableBonds`, etc.

---

## 🚀 Project Workflow

### 1. **Load the Data**

```python
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')
2. Data Preparation
Separate input features X and target variable y

Split dataset into training and testing sets (80/20 split)

python
Copy
Edit
from sklearn.model_selection import train_test_split

X = df.drop('logS', axis=1)
y = df['logS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
3. Model Building
🔹 Linear Regression
python
Copy
Edit
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
🔹 Random Forest Regressor
python
Copy
Edit
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)
4. Model Evaluation
Using metrics:

Mean Squared Error (MSE)

R-squared Score (R²)

python
Copy
Edit
from sklearn.metrics import mean_squared_error, r2_score

5. Model Comparison
python
Copy
Edit
df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)
This gives a quick overview of which model performs better on both training and test sets.

6. Prediction Visualization
We visualize how well the Linear Regression model predicts LogS:

python
Copy
Edit
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(x=y_train, y=y_lr_train_pred, c='#7CAE00', alpha=0.3)
z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), '#F8766D')
plt.xlabel("Experimental LogS")
plt.ylabel("Predicted LogS")
plt.title("Linear Regression: Experimental vs Predicted")
plt.show()
📌 Technologies Used
Python

Pandas

Scikit-learn

Matplotlib / NumPy

Google Colab

📈 Future Improvements
Hyperparameter tuning (e.g., GridSearchCV)

Add more models (e.g., XGBoost, SVR)

Feature selection or dimensionality reduction (PCA)

Save and load models (joblib/pickle)

📂 How to Run
Open Google Colab

Upload or link the first-project.ipynb

Run cells in sequence

Observe model results and plots

🙌 Acknowledgments
Dataset from Data Professor

Inspired by introductory ML tutorials in chemistry

yaml
Copy
Edit
