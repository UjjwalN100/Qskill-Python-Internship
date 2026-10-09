\# House Price Prediction



\## Project Overview

This project uses linear regression to estimate house sale prices from housing characteristics in the Kaggle House Prices dataset.



\## Features

\- Load housing data using Pandas

\- Handle missing numerical and categorical values

\- Encode neighborhood categories

\- Split the dataset into training and testing sets

\- Train a linear regression model

\- Evaluate predictions using MAE, RMSE, and R²

\- Save the trained model

\- Predict a house price from user-provided inputs



\## Technologies Used

\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Joblib



\## Dataset

The project uses the Kaggle House Prices dataset. It contains 1,460 housing records and 81 columns, including the SalePrice target.



\## How to Run

From the main internship folder, activate the virtual environment and install dependencies:



`python -m pip install -r requirements.txt`



Train the model:



`python Task3\_House\_Price\_Prediction/train\_model.py`



Run the prediction program:



`python Task3\_House\_Price\_Prediction/predict\_price.py`



\## Model Evaluation

Results from the recorded test run:

\- Mean Absolute Error: 22,265.31

\- Root Mean Squared Error: 36,076.50

\- R-squared: 0.8303



\## Limitations

Predictions depend on the training dataset and selected features. They are estimates, not guaranteed market valuations.



\## Internship

QSkill Python Development Internship — September–October 2026.

