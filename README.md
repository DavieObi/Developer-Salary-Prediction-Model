# 💻 Developer Salary Prediction Model

-----

## Overview

This repository contains a machine learning project for **predicting developer salaries** based on key factors like country, education level, and years of professional coding experience. The core of the project is a Jupyter notebook (`salary backup.ipynb`) that covers data cleaning, exploratory data analysis, feature engineering, model training, and saving the final model for deployment.

## Key Features

  * **Data Cleaning and Preparation:** Comprehensive handling of missing values, filtering for full-time professional developers, and outlier removal.
  * **Feature Engineering:** Categorical features like `Country` and `EdLevel` are processed using a custom function to group low-frequency categories into 'Others'. The `YearsCodePro` is converted to a numeric format, and the `Salary` column is log-transformed for better model training.
  * **Model Training:** Evaluates multiple regression models, including **Random Forest Regressor**, Decision Tree Regressor, and Linear Regression.
  * **Model Persistence:** The best-performing model (Random Forest) and the necessary data transformers (Label Encoders) are saved using `joblib` for easy loading and prediction in a production environment.

## Data Source

The model is built using the **Stack Overflow Developer Survey** results, which provides a comprehensive dataset on developers worldwide. Specifically, the model targets the `ConvertedComp` (renamed to `Salary`) column, which represents the annual salary converted to USD.

## Prerequisites

To run the notebook and use the model, you need the following libraries:

  * Python (3.x)
  * `pandas`
  * `numpy`
  * `matplotlib`
  * `scikit-learn`
  * `joblib`

You can install the required packages using pip:

```bash
# This is based on the libraries used in the notebook
pip install pandas numpy matplotlib scikit-learn joblib
```

## Project Files

  * `salary backup.ipynb`: The main Jupyter notebook containing all the steps from data loading and preparation to model training and saving.
  * `survey_results_public.csv`: The raw dataset used for training (not included in this repository structure, but needed to run the notebook).
  * `saved_steps.joblib`: The trained machine learning model (Random Forest Regressor) and the fitted `LabelEncoder` objects for the `Country` and `EdLevel` columns.

## Model Performance

The **Random Forest Regressor** was selected as the final model, achieving the highest performance with an R-squared score of **0.7107**.

## Usage (Making Predictions)

The model and its transformers are saved in `saved_steps.joblib`. You can load this file to make new predictions without re-running the entire training process:

```python
import joblib
import numpy as np

# Load the saved data
data = joblib.load("saved_steps.joblib")

# Extract the model and encoders
regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# Example Prediction Function
def predict_salary(country, education_level, years_pro):
    # 1. Transform categorical inputs using the loaded encoders
    country_encoded = le_country.transform([country])[0]
    education_encoded = le_education.transform([education_level])[0]
    
    # 2. Convert years to the format expected by the model (e.g., float)
    if years_pro == 'Less than 1 year':
        years_pro = 0.5
    else:
        years_pro = float(years_pro)

    # 3. Create the input array
    X = np.array([[country_encoded, education_encoded, years_pro]])
    
    # 4. Predict the log-transformed salary
    log_salary = regressor_loaded.predict(X)
    
    # 5. Inverse transform the prediction to get the actual USD salary
    salary = np.exp(log_salary[0])
    
    return salary

# Example Usage
# This example input is inferred from the model's test prediction
# predict_salary("United States", "Bachelor's degree (B.A., B.S., B.Eng., etc.)", 15) 
# returns array([145332.08080808])
```

-----

*Note: For the above prediction example to work, the input strings for `country` and `education_level` must exactly match the labels seen during training, or they must be mapped to the 'Others' category if they were not one of the top categories.*
