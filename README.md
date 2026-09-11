# 🏠 RealtySense — Data Driven Property Valuation

A data-driven machine learning project for predicting residential property prices based on key property characteristics such as area, bedrooms, bathrooms, stories, parking, location-related attributes, and other amenities.

The project uses **Python, Pandas, Seaborn, Statsmodels, and Scikit-learn** to perform exploratory data analysis, feature preprocessing, feature selection, regression modelling, and model evaluation.

---

## 📌 Project Overview

Real estate prices are influenced by multiple property characteristics, making data-driven valuation useful for buyers, sellers, and investors.

This project analyzes **545 residential property records with 13 key features** to identify the factors that influence property prices and build a predictive model for estimating property values.

### Objectives

* Explore and understand the housing dataset.
* Identify relationships between property characteristics and prices.
* Convert categorical variables into machine-readable features.
* Scale numerical features for modelling.
* Analyze feature correlation and multicollinearity.
* Select relevant features using statistical analysis and VIF.
* Build a multiple linear regression model.
* Analyze model residuals.
* Evaluate the model on unseen test data using **R² score**.

---

## 📊 Dataset

The project uses the **Housing.csv** dataset containing **545 property records**.

### Features

| Feature            | Description                                        |
| ------------------ | -------------------------------------------------- |
| `area`             | Property area                                      |
| `bedrooms`         | Number of bedrooms                                 |
| `bathrooms`        | Number of bathrooms                                |
| `stories`          | Number of stories                                  |
| `mainroad`         | Whether the property is connected to the main road |
| `guestroom`        | Whether the property has a guest room              |
| `basement`         | Whether the property has a basement                |
| `hotwaterheating`  | Whether hot water heating is available             |
| `airconditioning`  | Whether air conditioning is available              |
| `parking`          | Number of parking spaces                           |
| `prefarea`         | Whether the property is in a preferred area        |
| `furnishingstatus` | Furnishing status of the property                  |
| `price`            | Target property price                              |

---

## 🔎 Exploratory Data Analysis

The dataset was initially explored using:

* Dataset shape and structure
* Descriptive statistics
* Pair plots
* Box plots
* Correlation analysis

Visualizations were used to understand how numerical and categorical features relate to property prices.

Particular attention was given to features such as:

* Property area
* Bedrooms
* Bathrooms
* Location/preferred area
* Property amenities

---

## 🧹 Data Preprocessing

### Binary Encoding

Binary `yes/no` variables were converted into numerical values:

```text
yes → 1
no  → 0
```

The following variables were encoded:

* `mainroad`
* `guestroom`
* `basement`
* `hotwaterheating`
* `airconditioning`
* `prefarea`

### One-Hot Encoding

The `furnishingstatus` categorical variable was converted into dummy variables using one-hot encoding.

The original categorical column was then removed.

### Train-Test Split

The dataset was divided into:

* **70% training data**
* **30% testing data**

using `train_test_split` with a fixed random state for reproducibility.

### Feature Scaling

Numerical variables were normalized using **MinMaxScaler**.

Scaled variables include:

```text
area
bedrooms
bathrooms
stories
parking
price
```

---

## 🤖 Model Development

The project uses **Multiple Linear Regression** implemented using `statsmodels`.

The modelling process was developed iteratively.

### 1. Single Feature Model

The initial model was trained using:

```text
area
```

### 2. Feature Expansion

Additional features were progressively introduced, including:

```text
area
bathrooms
bedrooms
```

### 3. Full Model

The model was then expanded to include all available predictors.

### 4. Feature Selection

Feature selection was performed using:

* Statistical significance from the regression summary
* Correlation analysis
* Variance Inflation Factor (VIF)

Features were removed iteratively to reduce multicollinearity and improve the model.

The final modelling stage removed:

```text
semi-furnished
bedrooms
```

---

## 📈 Multicollinearity Analysis

**Variance Inflation Factor (VIF)** was used to identify highly correlated predictors.

The VIF analysis helped determine whether individual features were providing independent information or introducing multicollinearity into the regression model.

This allowed the model to be simplified while retaining important predictive variables.

---

## 📉 Residual Analysis

Residuals were calculated as:

```text
Residual = Actual Price − Predicted Price
```

The distribution of residuals was visualized to assess the behaviour of prediction errors and check whether the model showed major systematic deviations.

---

## 🧪 Model Evaluation

The final model was evaluated on the **unseen test dataset**.

The primary evaluation metric used in the notebook is:

### R² Score

R² measures the proportion of variation in property prices explained by the model.

```python
from sklearn.metrics import r2_score

r2_score(y_true=y_test, y_pred=y_test_pred)
```

> **Note:** The project should report the exact R² value obtained from the notebook rather than referring to it as “accuracy,” since this is a regression problem.

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries

* **Pandas** — Data manipulation and preprocessing
* **NumPy** — Numerical operations
* **Matplotlib** — Data visualization
* **Seaborn** — Statistical visualization
* **Scikit-learn** — Data preprocessing, train-test split and evaluation
* **Statsmodels** — Ordinary Least Squares regression and statistical analysis

---

## 📂 Project Structure

```text
Real-Estate-Price-Prediction/
│
├── Real Estate Price Prediction.ipynb
├── Housing.csv
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Real-Estate-Price-Prediction
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
```

### 3. Open the notebook

```bash
jupyter notebook
```

Open:

```text
Real Estate Price Prediction.ipynb
```

### 4. Run all cells

Make sure `Housing.csv` is available at the expected dataset path or update the `pd.read_csv()` path in the notebook.

---

## 💡 Key Insights

The analysis focuses on identifying the property characteristics that have the strongest relationship with house prices.

The modelling process highlights the importance of factors such as:

* **Area**
* **Bathrooms**
* **Location/preferred-area characteristics**

while statistical feature-selection techniques help remove less useful or highly correlated variables.

---

## 📌 Future Improvements

The current project uses multiple linear regression. It could be extended by:

* Comparing Linear Regression with Random Forest, Gradient Boosting, and XGBoost.
* Using cross-validation for more robust model evaluation.
* Performing hyperparameter tuning for tree-based models.
* Adding MAE and RMSE alongside R².
* Building an interactive property-price prediction dashboard using Streamlit.
* Adding location-based features for more granular valuation.
* Deploying the final model as a web application or API.

---

## 👩‍💻 Author

**Suhani Gupta**

B.Tech — Petroleum Engineering
IIT (ISM) Dhanbad

---

## ⭐ Project Highlights

* Analyzed **545 residential property records**
* Performed exploratory data analysis and visualization
* Applied binary and one-hot encoding
* Used **Min-Max feature scaling**
* Applied **VIF-based multicollinearity analysis**
* Built an **OLS multiple linear regression model**
* Performed residual analysis
* Evaluated predictions using **R² score**
