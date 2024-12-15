
from sklearn.tree import DecisionTreeClassifier
import scipy.sparse as sp
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import RocCurveDisplay
from prettytable import PrettyTable
df = pd.read_csv("prompts.csv")
equal = df["options argued for"] == df["true answer"]
equal = equal.tolist()
equal = np.array(equal)
mat = np.load("all_sparse_mats.npy",allow_pickle=True)
mat = sp.vstack(mat)






# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(mat, equal, test_size=0.2, random_state=42)

# Train a Decision Tree classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train.toarray(), y_train)  # Convert sparse matrix to dense for Decision Tree

# Predictions
y_train_pred = classifier.predict(X_train.toarray())
y_test_pred = classifier.predict(X_test.toarray())
y_test_proba = classifier.predict_proba(X_test.toarray())[:, 1]

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

# Print metrics using PrettyTable
table = PrettyTable()
table.field_names = ["Metric", "Train", "Test"]
table.add_row(["Accuracy", train_accuracy, test_accuracy])
table.add_row(["AUC", "-", test_auc])
print(table)

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

# Plot ROC Curve
RocCurveDisplay.from_predictions(y_test, y_test_proba)
plt.title('ROC Curve')
plt.show()

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
