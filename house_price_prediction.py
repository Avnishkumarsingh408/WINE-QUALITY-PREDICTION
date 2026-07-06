import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("Housing.csv")

# Display dataset
print("Dataset:")
print(data)

# Features
X = data[['Area', 'Bedrooms', 'Age']]

# Target
y = data['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("\nModel Evaluation")
print("----------------------")
print("R2 Score :", r2_score(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))

# Predict new house
print("\nPrediction Example")

area = 2400
bedrooms = 4
age = 2

price = model.predict([[area, bedrooms, age]])

print(f"Area = {area}")
print(f"Bedrooms = {bedrooms}")
print(f"Age = {age}")
print("Predicted Price = ₹{:,.2f}".format(price[0]))

# Plot Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Price")
plt.grid(True)
plt.show()