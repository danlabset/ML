# California Housing Price Prediction ML with XGBOOST , Linera Regression and  RandomForestRegressor 

## Overview
This repository contains a Python script for predicting housing prices in California using the California Housing dataset. The script utilizes the XGBoost regression model for training and evaluation.

## Dependencies
Ensure you have the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Dataset
The California Housing dataset is loaded using scikit-learn's fetch_california_housing function. The dataset contains information about median income, house age, average rooms, average bedrooms, population, average occupancy, latitude, and longitude, with the target variable being the median house value.

## Data Preprocessing
The dataset is loaded into a Pandas DataFrame, and basic exploratory data analysis is performed. Duplicate records are removed, and missing values are checked (no missing values found).

## Exploratory Data Analysis (EDA)
Descriptive statistics and correlation analysis are conducted to understand the relationships between features. A heatmap is used to visualize the correlation matrix.

## Model Training
The dataset is split into training and test sets. The XGBoost regression model is trained on the training set.

## Model Evaluation
The model is evaluated using R-squared and Mean Squared Error (MSE) on both the training and test sets.

## Results
The XGBoost model shows promising performance with an R-squared of approximately 0.9446 on the training set and 0.9854 on the test set. The MSE on the test set is around 0.0191.

Linear Regression:
## Data Preprocessing:

The dataset is loaded from the California Housing dataset.
Features are selected, and the target variable (price_y) is separated.
Duplicate records are removed from the dataset.
## Model Training:

The data is split into training and test sets using the train_test_split function.
Linear Regression is applied to the training set using the fit method.
## Evaluation:

R-squared and Mean Squared Error (MSE) are calculated for the training set.
The computed MSE is compared to the square of the mean of the target variable to assess model performance.
Results:

MSE was 0.5 on test and Train set not peforming well. 

##Data Preprocessing:
Similar data preprocessing steps as in Linear Regression.

#Model Training:

The data is split into training and test sets using the train_test_split function.
A Random Forest model is instantiated and trained on the training set using the fit method.
##Evaluation:

R-squared and Mean Squared Error (MSE) are calculated for both the training and test sets.
The results indicate a Cross-Validation MSE value, suggesting the use of cross-validation for a more robust evaluation.
## Results:

The Random Forest model shows improved performance on the test set compared to the Linear Regression model.
Cross-Validation MSE is lower than the initial training set MSE, indicating potential mitigation of overfitting.

Additional Considerations:
The Random Forest model seems to outperform the Linear Regression model, as indicated by lower MSE values on the test set.
Hyperparameter tuning and cross-validation are utilized to enhance the Random Forest model's generalization.
Further improvements could involve exploring additional features, regularization, or trying other regression algorithms.
These summaries provide an overview of the Linear Regression and Random Forest models in the context of the provided code. For a more detailed analysis, it would be helpful to review the complete code and data.

## Visualization
A scatter plot is created to visualize the actual prices against the predicted prices on the training set.

Feel free to explore and modify the script for further experimentation and improvement.

You can install them using the following command:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost



