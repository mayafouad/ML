# 1. 🎓 Binary Logistic Regression – University Admission Prediction 
Suppose you are the administrator of a university department and you want
to determine each applicant's chance of acceptance based on their results on
two exams. You have data from previous applicants which consists of the
applicants' scores on two exams and the admissions decision.

### ⚙️ Steps Implemented

#### 1. Data Preprocessing
Load dataset containing:
Exam 1 score & Exam 2 score

Decision (1 = admitted, 0 = not admitted)
```python
X = data.drop(columns=['decision']).round(2)
y = data['decision']
```

Split the dataset into training and testing sets:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=100)
```

#### 2. Feature Scaling (Min-Max Normalization)

To ensure both exam scores have the same influence, we scale them to the [0, 1] range:
```python
xmin = X_train.min()
xmax = X_train.max()
X_train = (X_train - xmin) / (xmax - xmin)
X_test = (X_test - xmin) / (xmax - xmin)
```

This step improves model convergence and accuracy.

#### 3. Visualize Feature Distribution

Boxplots before and after scaling are plotted to verify normalization.
```python
X_train.plot(kind='box', subplots=True, sharey=True)
plot.show()
```

#### 4. Model Training

Train a Logistic Regression classifier using the SAG (Stochastic Average Gradient) solver for efficiency:
```python
from sklearn import linear_model
model = linear_model.LogisticRegression(solver='sag')
model.fit(X_train, y_train)
```

#### 5. Model Evaluation

Predict on the test set and evaluate accuracy:
```python
y_pred = model.predict(X_test)
print('Correct predictions ratio: %.2f' % model.score(X_test, y_test))
```

✅ Correct predictions ratio: 0.88

This means the model correctly predicts admission outcomes 88% of the time.

📈 Results Summary
Metric	Value
Accuracy	0.88
Algorithm	Logistic Regression
Solver	SAG
Features	Exam 1, Exam 2
Scaled	Yes (Min-Max)

#### 🧠 Insights

Logistic Regression effectively models the probability of admission.
Feature scaling significantly stabilizes model training.
With only two exam scores, the model still achieves high accuracy (88%). 


---

# 2. 🌸 Multi-Class Logistic Regression on Iris Dataset – 2D Visualization
This project demonstrates how to train and visualize a Logistic Regression classifier on the classic Iris dataset, focusing on two features: sepal length and sepal width.
The model predicts the flower class (Setosa, Versicolor, or Virginica) and visualizes the decision boundaries between classes in a 2D plot.

## 📘 Overview

Logistic Regression is a linear model used for classification problems.
In this example, we:
1. Load and preprocess the Iris dataset.
2. Train a Logistic Regression classifier.
3. Evaluate the model using accuracy and a classification report.
4. Visualize the model’s decision boundaries in 2D.

#### 📊 Dataset
The Iris dataset contains 150 samples with 4 features for each iris flower:
Feature	Description:
Sepal length	Length of the sepal (cm)
Sepal width	Width of the sepal (cm)
Petal length	Length of the petal (cm)
Petal width	Width of the petal (cm)

Each sample belongs to one of three classes:
🌺 Iris-setosa
🌿 Iris-versicolor
🌸 Iris-virginica

In this project, only sepal length and sepal width are used for visualization purposes.

#### ⚙️ Steps Implemented
1. Load the Dataset
```python
data = pandas.read_csv('iris.txt', names=['sepal-length', 'sepal-width','petal-length', 'petal-width', 'class'])
```
2. Select Features
Only the first two features are used:
```python
X = data[['sepal-length', 'sepal-width']].round(2)
y = data['class']
```

3. Split the Data
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

4. Train the Logistic Regression Model
```python
model = LogisticRegression(solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)
```

5. Evaluate Model Performance
```python
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

##### Results:
Accuracy: 0.84
Classification Report:
                 precision    recall  f1-score   support
    Iris-setosa       1.00      1.00      1.00        15
Iris-versicolor       0.78      0.64      0.70        11
 Iris-virginica       0.71      0.83      0.77        12

### 🎨 Visualization

The code visualizes decision boundaries that separate different classes using matplotlib.
The background colors represent the regions classified as Setosa, Versicolor, or Virginica.
Each data point is plotted with its true class color.
```python
plt.contourf(xx, yy, Z_int, alpha=0.3, cmap=cmap)
plt.scatter(X['sepal-length'], X['sepal-width'], c=..., edgecolors='k', s=50)
```

Plot Example:
Red → Iris-setosa
Green → Iris-versicolor
Blue → Iris-virginica

##### 📈 Final Output

Accuracy: 84%
Visualization: Decision boundaries clearly separate Setosa, while Versicolor and Virginica show partial overlap due to feature similarity.

