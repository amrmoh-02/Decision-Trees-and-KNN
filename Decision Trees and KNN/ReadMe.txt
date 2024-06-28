Decision Trees and KNN
Dataset Description
The dataset "BankNote_Authentication.csv" contains four features (variance, skew, curtosis, and entropy) and a class attribute indicating whether a banknote is real or forged.

Problem 1: Decision Trees using Scikit-learn
Using the Banknote Authentication dataset, we implemented and experimented with decision trees.

Fixed Train-Test Split Ratio (25%-75%):

Conducted the experiment five times with different random splits.
Recorded the sizes and accuracies of the trees for each run.
Variable Train-Test Split Ratios:

Conducted experiments with split ratios of 30%-70%, 40%-60%, 50%-50%, 60%-40%, and 70%-30%.
For each ratio, the experiment was repeated with five different random seeds.
Calculated and printed the mean, maximum, and minimum accuracy for each split ratio.
Calculated and printed the mean, max, and min tree sizes for each split ratio.
Plotted:
Mean accuracy against training set size.
Mean number of nodes in the final tree against training set size.
Problem 2: KNN
Using the Banknote Authentication dataset, we implemented a simple KNN classifier from scratch in Python.

Data Split:

Divided the data into 70% for training and 30% for testing.
Normalization:

Normalized each feature column separately using the mean and standard deviation from the training data.
KNN Implementation:

Implemented the KNN algorithm using Euclidean distance.
Handled ties by selecting the class that appears first in the training file.
Experiments with Different k Values:

Conducted experiments with k values from 1 to 9.
Printed:
The value of k used.
The number of correctly classified test instances.
The total number of instances in the test set.
The accuracy.