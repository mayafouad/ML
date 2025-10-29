# 🏠💰 House Price Prediction using Multiple Linear Regression

## 📘 Overview  
Ever wondered what factors determine a **house’s price**? 🏡  
This project uses **Multiple Linear Regression (MLR)** to predict housing prices based on various **numerical and categorical features**.  

The model learns from historical housing data and estimates prices considering factors like **area**, **bedrooms**, **bathrooms**, **stories**, **location access**, and **furnishing status**.

---

## 🚀 Project Workflow  

### 1️⃣ Explore and Visualize the Target Variable  

Visualize how house prices are distributed in the dataset:

```python
plt.figure(figsize=(8, 6))
plt.title('House Price Distribution Plot')
sns.boxplot(y=df['price'])
plt.ylabel('Price')
plt.show()
```

### 2️⃣ Define Features and Target

Split the dataset into:

X (Features): all columns except price
y (Target): the price column
```python
X = df.drop('price', axis=1)
y = df['price']
```

### 3️⃣ Identify Feature Types
```python
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                    'airconditioning', 'prefarea', 'furnishingstatus']
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
```

### 4️⃣ Preprocessing Pipeline

1. Scale numerical features using MinMaxScaler
2. Encode categorical features using OneHotEncoder
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

```
We also scale target prices for better regression performance:
```python
y = MinMaxScaler().fit_transform(y.values.reshape(-1, 1)).ravel()
```

### 5️⃣ Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```

Apply preprocessing:
```python
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
```

### 6️⃣ Train the Model
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

7️⃣ Model Evaluation

Make predictions and evaluate using Mean Squared Error (MSE):
```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
```

Output:

Mean Squared Error (MSE): 0.01

### 8️⃣ Feature Importance

Identify which features most influence house prices:
```python
feature_names = (
    numerical_cols + 
    preprocessor.named_transformers_['cat']
    .get_feature_names_out(categorical_cols).tolist()
)

coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_
})
print(coefficients)
```

Example Output:

Feature	Coefficient
area	0.2973
bedrooms	0.0332
bathrooms	0.2843
stories	0.1058
parking	0.0584
mainroad_yes	0.0319
guestroom_yes	0.0201
basement_yes	0.0338
hotwaterheating_yes	0.0593
airconditioning_yes	0.0685
prefarea_yes	0.0545
furnishingstatus_semi-furnished	-0.0110
furnishingstatus_unfurnished	-0.0358


### 9️⃣ Visualization
```python
# 📊 Actual vs Predicted Prices
plt.scatter(y_test, y_pred, color='green')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

#📉 Residual Plot
# To check model assumptions and fit quality:
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, color='blue', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

📊 Example Results
Metric	Value
Mean Squared Error (MSE)	0.01
R² Score	≈ 0.98

📈 The model explains nearly 98% of the variation in house prices — indicating a strong predictive performance.