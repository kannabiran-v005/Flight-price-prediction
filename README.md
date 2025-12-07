Flight Price Prediction Using Random Forest Regression
This project focuses on predicting flight ticket prices using a Random Forest Regression model. The dataset includes flight-related information such as airline, cities, duration, number of stops, travel class, and days left before the journey. After performing data preprocessing and one-hot encoding, the model learns to approximate real-world flight prices with high accuracy.

The workflow includes cleaning the dataset, encoding categorical features, splitting the data into training and testing sets, training a Random Forest model with tuned hyperparameters (n_estimators=290, max_depth=37), and evaluating its performance using MAE and RMSE metrics.

Additionally, the trained model can take new flight details as user input and predict the ticket price, making it suitable for real-time applications and travel cost prediction systems.

The model achieves the following performance metrics on the test set:

MAE: 1056.42

RMSE: 2709.03
