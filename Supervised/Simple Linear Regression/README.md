# 🏙️💰 Predicting Profits for a New Store using Linear Regression
### 📘 Overview
    Suppose you’re the CEO of a big company considering different cities for a new store 🏪.
    You already have historical data for existing cities — population vs. profit.
    In this project, we use Linear Regression to predict the expected profit for a new city based on its population.

### 🚀 Project Workflow
#### 1️⃣ Load and Prepare the Data
    The dataset contains two columns:
    population — population of the city
    profit — profit from the store in that city

    Data is shuffled to ensure randomness and split into training and testing sets.


#### 2️⃣ Train the Model
    We use Linear Regression from sklearn.linear_model to train our model.
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)

    Output Example:
    Coefficients: [[1.23074787]]   Intercept: [-4.25715253]
    This means:
    Profit ≈ 1.23 × Population - 4.26

#### 3️⃣ Evaluate the Model
    We predict on the test set and measure the Mean Squared Error (MSE).

    y_pred = model.predict(x_test)
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

    Example Output:
    Mean squared error: 6.41

#### 4️⃣ Predict for New Cities 🌆
    Now we load a new dataset (cities.csv) containing city names and populations, then predict their expected profits.

    new_pop_data = pandas.read_csv("cities.csv", names=['city','population'])
    new_data = new_pop_data['population'].to_numpy().reshape(-1,1)
    predictions = model.predict(new_data)
