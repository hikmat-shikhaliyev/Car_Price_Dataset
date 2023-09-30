# Car_Price_Dataset
This repository contains Python code for predicting car prices based on various features. The code uses a dataset (CarPrice_Assignment.csv) containing information about cars, such as horsepower, city mileage per gallon (citympg), drive wheel type (drivewheel), cylinder number, and fuel system, among others. The code employs linear regression for the prediction task.
# Code Overview
car_price_prediction.py: The main Python script containing the code for data preprocessing, feature selection, model training, and evaluation. It uses linear regression to predict car prices based on selected features.
# Data Preprocessing
Handling Missing Values: The script checks for missing values in the dataset and handles them appropriately. Feature Selection: The script drops irrelevant features (CarName, car_ID, etc.) and selects relevant features for the prediction task. Outlier Detection and Removal: Outliers are detected using the Interquartile Range (IQR) method and removed from the dataset. Log Transformation: The target variable (price) is log-transformed for better prediction accuracy.
# Model Training and Evaluation
Feature Scaling: Features are standardized using the StandardScaler from scikit-learn. Model Training: The linear regression model is trained using the scaled features. Model Evaluation: The script evaluates the model's performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) score on both training and test datasets. Feature Importance: The importance of individual features is evaluated based on their impact on the R-squared score.
# Results
The script outputs the evaluation metrics for both the training and test datasets. Additionally, it provides insights into the importance of individual features for the prediction task.
# Conclusion
The code demonstrates a simple yet effective approach to predict car prices using linear regression.
