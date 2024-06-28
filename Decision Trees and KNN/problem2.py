import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Problem 2
# Define KNN function
def knn(train_data, test_data, k):
    predictions = []
    for test_instance in test_data:
        distances = []
        for idx, train_instance in enumerate(train_data):
            distance = np.sqrt(np.sum((test_instance[:-1] - train_instance[:-1]) ** 2))
            # calculates the distance between the feature vectors of current test instance and training instance.
            # It excludes the last element of each vector (class label)
            distances.append((distance, idx, train_instance[-1]))  # Add idx to retain the original indices
        
        # Sort distances and select k nearest neighbors
        sorted_distances = sorted(distances)
        if sorted_distances:  # Check if sorted_distances is not empty
            nearest_neighbors = sorted_distances[:k]
            neighbors = [train_data[neighbor[1]][-1] for neighbor in nearest_neighbors]  # Access class label using original index
            # It appends  the distance, index of the training instance, and its class label to the distances list.
            
            if neighbors:  # Check if neighbors list is not empty
                # Count the class votes
                vote = Counter(neighbors).most_common(1)[0][0]
                predictions.append(vote)
            else:
                print("Warning: No neighbors found for the test instance.")
        else:
            print("Warning: sorted_distances is empty.")
    
    return predictions

# Load the dataset
data = pd.read_csv("BankNote_Authentication.csv")

# Normalize the data
normalized_data = (data - data.mean()) / data.std()

# Split data into train and test sets
train_size = int(0.7 * len(data))
train_data = normalized_data[:train_size].values
test_data = normalized_data[train_size:].values

# Run KNN for different values of k
for k in range(1, 10):
    predictions = knn(train_data, test_data, k)
    correct_predictions = sum(predictions == test_data[:,-1])
    total_instances = len(test_data)
    accuracy = correct_predictions / total_instances
    
    print(f"K = {k}:")
    print(f"Correctly classified instances: {correct_predictions}")
    print(f"Total instances in test set: {total_instances}")
    print(f"Accuracy: {accuracy}")
    print()
