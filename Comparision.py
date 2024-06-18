import matplotlib.pyplot as plt
import numpy as np

# Evaluation metrics for all three models

models = ['1D CNN', 'RNN', 'LSTM']

accuracy = [0.839, 0.780, 0.811]
precision = [0.838, 0.788, 0.794]
recall = [0.844, 0.773, 0.845]
f1_score = [0.841, 0.781, 0.819]
auc_roc = [0.838, 0.896, 0.811]

# Plotting bar charts

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Accuracy
axes[0, 0].bar(models, accuracy, color=['blue', 'orange', 'green'])
axes[0, 0].set_title('Accuracy')
axes[0, 0].set_ylim(0.7, 1)  # Adjust the y-axis limits if necessary

# Precision
axes[0, 1].bar(models, precision, color=['blue', 'orange', 'green'])
axes[0, 1].set_title('Precision')
axes[0, 1].set_ylim(0.7, 1)  # Adjust the y-axis limits if necessary

# Recall
axes[1, 0].bar(models, recall, color=['blue', 'orange', 'green'])
axes[1, 0].set_title('Recall')
axes[1, 0].set_ylim(0.7, 1)  # Adjust the y-axis limits if necessary

# F1 Score
axes[1, 1].bar(models, f1_score, color=['blue', 'orange', 'green'])
axes[1, 1].set_title('F1 Score')
axes[1, 1].set_ylim(0.7, 1)  # Adjust the y-axis limits if necessary

plt.tight_layout()
plt.show()

# Plotting AUC-ROC
plt.figure(figsize=(8, 6))
plt.bar(models, auc_roc, color=['blue', 'orange', 'green'])
plt.title('AUC-ROC')
plt.ylim(0.7, 1)  # Adjust the y-axis limits if necessary
plt.show()
