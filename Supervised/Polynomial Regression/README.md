# 🍦📈 Predicting Ice Cream Sales using Polynomial Regression

## 📘 Overview  
Ever wondered how **temperature** affects **ice cream sales**? 🍨  
In this project, we use **Polynomial Regression** to predict the number of ice cream units sold based on the temperature (°C).  

Unlike simple linear regression, polynomial regression captures the **nonlinear relationship** between variables — ideal for scenarios like this, where sales don’t increase perfectly linearly with temperature.

---

## 🚀 Project Workflow  

### 1️⃣ Data Scaling  
Before training, we apply **Min-Max Scaling** to normalize both `Temperature` and `Sales` values between 0 and 1 for better model performance.

```python
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[['Temperature']])
y_scaled = scaler.fit_transform(df[['Sales']])
```
### 2️⃣ Polynomial Regression

We create polynomial features (degree = 2) to capture the curvature in the relationship.
```python
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
```

Then fit a Linear Regression model on these polynomial features:
```python
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
```

### 3️⃣ Model Prediction and Visualization

We generate smooth predictions across the temperature range to visualize the polynomial curve.
```python
X_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1,1)
X_range_poly = poly.transform(X_range)
y_poly_pred = poly_model.predict(X_range_poly)
```

##### 📉 Plotting:
```python
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_range.flatten(), y_poly_pred, color='red', linewidth=2, label='Polynomial fit (deg=2)')
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Cream Sales')
plt.title('Polynomial Regression Fit')
plt.legend()
plt.show()
```

### 4️⃣ Model Evaluation

We use Mean Squared Error (MSE) to evaluate how well the model fits the data.
```python
y_test_poly_pred = poly_model.predict(poly.transform(X_test))
mse = mean_squared_error(y_test, y_test_poly_pred)
print(f"Mean Squared Error (MSE): {mse}")
```

Example Output:
Mean Squared Error (MSE): 0.0038

### 5️⃣ Feature Coefficients

We display how much each polynomial feature contributes to the final prediction.
```python
feature_names = poly.get_feature_names_out()
coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': poly_model.coef_[0]
})
print(coefficients)
```

Example Output:
Feature	Coefficient
x0	2.85
x0²	-1.12

📊 Example Results
Temperature (°C)	Predicted Sales (units)
10	180
20	290
30	410
35	470

##### 📈 The model shows that ice cream sales rise with temperature — but the growth slows down at higher temperatures due to the nonlinear relationship.
