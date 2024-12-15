

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import RocCurveDisplay
from prettytable import PrettyTable
import scipy.sparse as sp
import seaborn as sns





df = pd.read_csv("prompts.csv")
equal = df["options argued for"] == df["true answer"]
equal = equal.tolist()
equal = np.array(equal)
mat = np.load("all_sparse_mats.npy",allow_pickle=True)
mat = sp.vstack(mat)
X_train, X_test, y_train, y_test = train_test_split(mat, equal, test_size=0.2, random_state=42)


# Define various alpha values for Lasso regression
alphas = [0.0000001,0.000001, 0.00001, 0.0001, 0.001]
results = []

# Train Lasso Logistic Regression for different alpha values
for alpha in alphas:
    classifier = LogisticRegression(penalty='l1', C=1/alpha, solver='saga', max_iter=1000)
    classifier.fit(X_train, y_train)

    # Predictions
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)
    y_test_proba = classifier.predict_proba(X_test)[:, 1]

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    # Store results
    results.append((alpha, train_accuracy, test_accuracy, test_auc))

# Print metrics using PrettyTable
table = PrettyTable()
table.field_names = ["Alpha", "Train Accuracy", "Test Accuracy", "Test AUC"]
for alpha, train_acc, test_acc, test_auc in results:
    table.add_row([alpha, train_acc, test_acc, test_auc])
print(table)

# Choose the best model based on AUC
best_model_idx = np.argmax([res[3] for res in results])
best_alpha = results[best_model_idx][0]
best_classifier = LogisticRegression(penalty='l1', C=1/best_alpha, solver='saga', max_iter=1000)
best_classifier.fit(X_train, y_train)

# Predictions for the best model
y_test_pred_best = best_classifier.predict(X_test)
y_test_proba_best = best_classifier.predict_proba(X_test)[:, 1]

# Confusion matrix for the best model
cm_best = confusion_matrix(y_test, y_test_pred_best)
print("Confusion Matrix for Best Model (Alpha = {:.3f}):".format(best_alpha))
print(cm_best)

# Classification report for the best model
print("Classification Report for Best Model:")
print(classification_report(y_test, y_test_pred_best))

# Plot ROC Curve for the best model
RocCurveDisplay.from_predictions(y_test, y_test_proba_best)
plt.title('ROC Curve for Best Model (Alpha = {:.3f})'.format(best_alpha))
plt.show()

# Plot Confusion Matrix for the best model
plt.figure(figsize=(6, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')





