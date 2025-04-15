from sklearn.metrics import average_precision_score

# True labels (ground truth) and predicted scores
y_true = [0, 1, 1, 0, 1, 1]  # Binary labels (0 or 1)
y_scores = [0.1, 0.4, 0.35, 0.8, 0.65, 0.9]  # Model's estimated scores

# Calculate average precision-recall score
average_precision = average_precision_score(y_true, y_scores)

# Print the result
print(f'Average precision-recall score: {average_precision}')
