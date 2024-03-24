import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Loading Dataset into Pandas DataFrame 'train_data', nuking the ID column
train_data = pd.read_csv('./KNN Algorithm/ExercícioKNN/Dados_Originais_11Features/TrainingData_11F_Original.txt', delimiter='\t', index_col=False)

# Drop the ID column
train_data = train_data.iloc[:, 1:]

# Separate features and target variable
x_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1:].values.ravel()

# print("X_TRAIN:", x_train, "\nY_TRAIN", y_train)

test_data = pd.read_csv('./KNN Algorithm/ExercícioKNN/Dados_Originais_11Features/TestingData_11F_Original.txt', delimiter='\t', index_col=False)

# Drop the ID column
test_data = test_data.iloc[:, 1:]

# Separate features and target variable
x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1:].values.ravel()

# print("X_TEST:", x_test, "\nY_TEST", y_test)

# Scatter plot of total.sulfur.dioxide and citric.acid for train_data

plt.figure(figsize=(10, 6))

# Plotting training data
for i in range(len(train_data)):
    if train_data.iloc[i]['class'] == 0:
        plt.scatter(train_data.iloc[i]['total.sulfur.dioxide'], train_data.iloc[i]['citric.acid'], 
                    color='blue', marker='o')
        plt.annotate('T{}'.format(i+1), (train_data.iloc[i]['total.sulfur.dioxide'], train_data.iloc[i]['citric.acid']),
                     textcoords="offset points", xytext=(5,-5), ha='right')
    else:
        plt.scatter(train_data.iloc[i]['total.sulfur.dioxide'], train_data.iloc[i]['citric.acid'], 
                    color='red', marker='o')
        plt.annotate('T{}'.format(i+1), (train_data.iloc[i]['total.sulfur.dioxide'], train_data.iloc[i]['citric.acid']),
                     textcoords="offset points", xytext=(5,-5), ha='right')

# Plotting test data
for i in range(len(test_data)):
    if test_data.iloc[i]['class'] == 0:
        plt.scatter(test_data.iloc[i]['total.sulfur.dioxide'], test_data.iloc[i]['citric.acid'], 
                    color='blue', marker='s')
        plt.annotate('N{}'.format(i+1), (test_data.iloc[i]['total.sulfur.dioxide'], test_data.iloc[i]['citric.acid']),
                     textcoords="offset points", xytext=(5,-5), ha='right')
    else:
        plt.scatter(test_data.iloc[i]['total.sulfur.dioxide'], test_data.iloc[i]['citric.acid'], 
                    color='red', marker='s')
        plt.annotate('N{}'.format(i+1), (test_data.iloc[i]['total.sulfur.dioxide'], test_data.iloc[i]['citric.acid']),
                     textcoords="offset points", xytext=(5,-5), ha='right')

plt.xlabel('total.sulfur.dioxide')
plt.ylabel('citric.acid')
plt.title('Quality of Wine')
plt.show()

# Number of Neighbours for Knn algorithm
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_7 = KNeighborsClassifier(n_neighbors=7)

knn_1.fit(x_train, y_train)
knn_3.fit(x_train, y_train)
knn_5.fit(x_train, y_train)
knn_7.fit(x_train, y_train)

# Use kneighbors to get both distances and indices of nearest neighbors
distances_1, indices_1 = knn_1.kneighbors(x_test)
distances_3, indices_3 = knn_3.kneighbors(x_test)
distances_5, indices_5 = knn_5.kneighbors(x_test)
distances_7, indices_7 = knn_7.kneighbors(x_test)

# Predictions
knn_1_pred = knn_1.predict(x_test)
knn_3_pred = knn_3.predict(x_test)
knn_5_pred = knn_5.predict(x_test)
knn_7_pred = knn_7.predict(x_test)

# Calculate accuracy
knn_accuracy_1 = round(accuracy_score(y_test, knn_1_pred))
knn_accuracy_3 = round(accuracy_score(y_test, knn_3_pred))
knn_accuracy_5 = round(accuracy_score(y_test, knn_5_pred))
knn_accuracy_7 = round(accuracy_score(y_test, knn_7_pred))

print("Accuracy 1-NN:\n", knn_accuracy_1 * 100)
print("Accuracy 3-NN:\n", knn_accuracy_3 * 100)
print("Accuracy 5-NN:\n", knn_accuracy_5 * 100)
print("Accuracy 7-NN:\n", knn_accuracy_7 * 100)

# Optionally, you can also print distances and indices for further analysis
print("Distances to nearest neighbors (1-NN):\n", distances_1)
print("Indices of nearest neighbors (1-NN):\n", indices_1)
print("Distances to nearest neighbors (3-NN):\n", distances_3)
print("Indices of nearest neighbors (3-NN):\n", indices_3)
print("Distances to nearest neighbors (5-NN):\n", distances_5)
print("Indices of nearest neighbors (5-NN):\n", indices_5)
print("Distances to nearest neighbors (7-NN):\n", distances_7)
print("Indices of nearest neighbors (7-NN):\n", indices_7)
