import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Define the training set
train_data = {
    'ID': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T24', 'T25', 'T26', 'T27', 'T28', 'T29', 'T30', 'T31', 'T32', 'T33', 'T34', 'T35', 'T36', 'T37', 'T38', 'T39', 'T40', 'T41', 'T42', 'T43', 'T44'],
    'total.sulfur.dioxide': [90, 110, 61, 77.5, 39, 18, 22, 18, 31, 71, 30, 60, 104, 43, 11, 34, 92, 9, 40, 37, 26, 52, 48, 63, 28, 34, 25, 12, 13, 25, 63, 43, 42, 51, 17, 78, 70, 54, 18, 21, 16, 61, 136, 40],
    'citric.acid': [0.38, 0.3, 0.41, 0, 0.35, 0.38, 0.6, 0.6, 0.26, 0.65, 0.02, 0.22, 0.32, 0.49, 0.02, 0.08, 0.2, 0.04, 0.15, 0.14, 0.4, 0.35, 0.39, 0, 0.24, 0.47, 0.01, 0.48, 0.49, 0.49, 0.25, 0.68, 0.05, 0.32, 0.42, 0.49, 0.23, 0.55, 0.23, 0.07, 0.45, 0.06, 0.28, 0.49],
    'class': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

# Define the test set
test_data = {
    'ID': ['N1', 'N2', 'N3', 'N4'],
    'total.sulfur.dioxide': [26, 72, 13, 32],
    'citric.acid': [0, 0.3, 0.5, 0.68],
    'class': [0, 0, 1, 1]
}

# print(len(train_data["total.sulfur.dioxide"]))
# print(len(train_data["ID"]))
# print(len(train_data["citric.acid"]))
# print(len(train_data["class"]))

# Convert dictionaries to dataframes
train_df = pd.DataFrame(train_data).set_index('ID')
test_df = pd.DataFrame(test_data).set_index('ID')

# Separate features and target variable
X_train = train_df[['total.sulfur.dioxide', 'citric.acid']]
y_train = train_df['class']

# Remove 'ID' column from test data
X_test = test_df[['total.sulfur.dioxide', 'citric.acid']]
y_test = test_df['class']

# Define the KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model
knn.fit(X_train, y_train)

# Predict the class of the test points
predicted_classes = knn.predict(X_test)

# Get the indices of the 5 nearest neighbors for each test point
indices_nearest_neighbors = knn.kneighbors(X_test, return_distance=False)

# Get the corresponding IDs
closest_IDs = [train_df.index[indices].tolist() for indices in indices_nearest_neighbors]

print("Closest T indexes to N4:", closest_IDs[3])  # Index 3 corresponds to N4