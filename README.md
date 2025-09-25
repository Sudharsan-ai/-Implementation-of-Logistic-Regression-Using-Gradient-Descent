## Implementation-of-Logistic-Regression-Using-Gradient-Descent
## AIM:

To write a program to implement Logistic Regression Using Gradient Descent.

## Equipments Required:

Hardware – PCs

Anaconda – Python 3.7 Installation / Jupyter Notebook

## Algorithm:

Import the required libraries.

Load the dataset.

Drop irrelevant columns.

Convert categorical columns into category type.

Encode categorical columns into numeric codes.

Define feature matrix X and target variable Y.

Initialize parameters.

Define the sigmoid function.

Define the loss function.

Define gradient descent for optimization.

Train the model using gradient descent.

Define the prediction function.

Make predictions.

Calculate model accuracy.

Test with new data.

## Program:
```
"""
Program to implement Logistic Regression Using Gradient Descent.
Developed by: SUDHARSAN S
RegisterNumber: 2122240403334
"""

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2. Load the Dataset
dataset = pd.read_csv("Placement_Data.csv")
print("\n--- Step 1: Original Dataset ---\n")
print(dataset.head())

# 3. Drop Irrelevant Columns
dataset = dataset.drop('sl_no', axis=1)
dataset = dataset.drop('salary', axis=1)

# 4. Convert Categorical Data to Category Type
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')

print("\n--- Step 2: Dataset Data Types ---\n")
print(dataset.dtypes)

# 5. Encode Categorical Data into Numerical Codes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes

print("\n--- Step 3: Encoded Dataset ---\n")
print(dataset.head())

# 6. Define Features (X) and Target (Y)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

print("\n--- Step 4: Target Variable (Y) ---\n")
print(Y)

# 7. Initialize Parameters
theta = np.random.randn(X.shape[1])
y = Y

# 8. Define Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 9. Define Loss Function
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# 10. Gradient Descent Function
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

# 11. Train the Model using Gradient Descent
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)

# 12. Prediction Function
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

# 13. Make Predictions on Training Data
y_pred = predict(theta, X)

print("\n--- Step 5: Predicted Values (y_pred) ---\n")
print(y_pred)

# 14. Calculate Model Accuracy
accuracy = np.mean(y_pred.flatten() == y)
print("\n--- Step 6: Model Accuracy ---\n")
print("Accuracy:", accuracy)

print("\n--- Step 7: Actual Target Values (Y) ---\n")
print(Y)

# 15. Test with New Data
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta, xnew)
print("\n--- Step 8: Prediction for New Input 1 ---\n")
print(y_prednew)

xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta, xnew)
print("\n--- Step 9: Prediction for New Input 2 ---\n")
print(y_prednew)

```
## Output:
Step 1: Original Dataset

![alt text](<Screenshot 2025-09-25 141318.png>)

Step 2: Dataset Data Types

![alt text](<Screenshot 2025-09-25 141324.png>)

Step 3: Encoded Dataset

![alt text](<Screenshot 2025-09-25 141331.png>)

Step 4: Target Variable (Y)

![alt text](<Screenshot 2025-09-25 141337.png>)

Step 5: Predicted Values (y_pred)

![alt text](<Screenshot 2025-09-25 142144.png>)

Step 6: Model Accuracy

![alt text](<Screenshot 2025-09-25 141348.png>)

Step 7: Actual Target Values (Y)

![alt text](<Screenshot 2025-09-25 141355.png>)

Step 8: Prediction for New Input 1

![alt text](<Screenshot 2025-09-25 141400.png>)

Step 9: Prediction for New Input 2

![alt text](<Screenshot 2025-09-25 141406.png>)

## Result:

Thus, the program to implement Logistic Regression Using Gradient Descent is written and verified using Python programming.
