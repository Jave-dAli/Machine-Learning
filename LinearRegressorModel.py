import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
# Define the mean squared error formula for later reference
# Mean squared error: sum((predicted - actual)^2) / n

# Load the diabetes dataset from scikit-learn
diabetes = datasets.load_diabetes()

# Display the keys in the dataset dictionary
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
diabetes.keys()

# Extract the third feature (column) from the dataset as the input variable
# if the comment from the followinng line is there, it'll be more accurate
# but then plotting will be complex, so comment out plt stuff
diabetes_X = diabetes.data # [:, np.newaxis, 2]

# Split the dataset into training and testing sets
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

# Create a linear regression model
model = linear_model.LinearRegression()

# Train the model using the training data
model.fit(diabetes_X_train, diabetes_y_train)

# Make predictions on the test data
diabetes_y_predicted = model.predict(diabetes_X_test)

# Print the mean squared error of the predictions
print("Mean squared error is:", mean_squared_error(diabetes_y_test, diabetes_y_predicted))

# Print the coefficients (weights) and the intercept of the linear regression model
print("Weights:", model.coef_)
print("Intercept:", model.intercept_)

# Plot the actual test data points and the predicted values
# plt.scatter(diabetes_X_test, diabetes_y_test)
# plt.plot(diabetes_X_test, diabetes_y_predicted, color='purple')
# plt.show()
