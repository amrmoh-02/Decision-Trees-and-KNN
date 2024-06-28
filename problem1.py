import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Problem 1
# Load the dataset
data = pd.read_csv("BankNote_Authentication.csv")

# 1. Experiment with a fixed train_test split ratio: Use 25% of the samples for training and the rest for testing.
# a. Run this experiment five times and notice the impact of different random splits of the data into training and test sets.
# b. Print the sizes and accuracies of these trees in each experiment.

for i in range(5):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1), data['class'], test_size=0.25, random_state=i)
    
    # Create Decision Tree Classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = clf.score(X_test, y_test)
    tree_size = clf.tree_.node_count
    
    print(f"Experiment {i+1}: Tree size: {tree_size}, Accuracy: {accuracy}")

# 2. Experiment with different range of train_test split ratio: Try (30%-70%), (40%-60%), (50%-50%), (60%-40%) and (70%-30%):
# a. Run the experiment with five different random seeds for each of split ratio.
# b. Calculate mean, maximum and minimum accuracy for each split ratio and print them.
# c. Print the mean, max and min tree size for each split ratio.
# d. Draw two plots:
# 1) shows mean accuracy against training set size
# 2) the mean number of nodes in the final tree against training set size.

split_ratios = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]
accuracies = []
tree_sizes = []

for ratio in split_ratios:
    accuracy_per_ratio = []
    tree_size_per_ratio = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1), data['class'], test_size=ratio[1], random_state=i)
        
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        
        accuracy = clf.score(X_test, y_test)
        tree_size = clf.tree_.node_count
        
        accuracy_per_ratio.append(accuracy)
        tree_size_per_ratio.append(tree_size)
    
    # Calculate mean, max, and min accuracy and tree size
    mean_accuracy = np.mean(accuracy_per_ratio)
    max_accuracy = np.max(accuracy_per_ratio)
    min_accuracy = np.min(accuracy_per_ratio)
    
    mean_tree_size = np.mean(tree_size_per_ratio)
    max_tree_size = np.max(tree_size_per_ratio)
    min_tree_size = np.min(tree_size_per_ratio)
    
    accuracies.append((mean_accuracy, max_accuracy, min_accuracy))
    tree_sizes.append((mean_tree_size, max_tree_size, min_tree_size))
    
    print(f"Split Ratio {ratio}:")
    print(f"Mean Accuracy: {mean_accuracy}, Max Accuracy: {max_accuracy}, Min Accuracy: {min_accuracy}")
    print(f"Mean Tree Size: {mean_tree_size}, Max Tree Size: {max_tree_size}, Min Tree Size: {min_tree_size}")
    print()

# Plotting
mean_accuracies = [acc[0] for acc in accuracies]
mean_tree_sizes = [size[0] for size in tree_sizes]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot([ratio[1] * len(data) for ratio in split_ratios], mean_accuracies, marker='o')
plt.title('Mean Accuracy vs Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Accuracy')

plt.subplot(1, 2, 2)
plt.plot([ratio[1] * len(data) for ratio in split_ratios], mean_tree_sizes, marker='o')
plt.title('Mean Tree Size vs Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Tree Size')

plt.tight_layout()
plt.show()
