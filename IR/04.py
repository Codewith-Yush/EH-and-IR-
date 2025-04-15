def calculate_metrics(retrieved_set, relevant_set):
    # Calculate true positives, false positives, and false negatives
    true_positive = len(retrieved_set.intersection(relevant_set))
    false_positive = len(retrieved_set.difference(relevant_set))
    false_negative = len(relevant_set.difference(retrieved_set))

    # Print the metrics
    print("True Positive: ", true_positive)
    print("False Positive: ", false_positive)
    print("False Negative: ", false_negative)
    print()

    # Calculate precision, recall, and F-measure
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_measure


# Example sets
retrieved_set = set(["doc1", "doc2", "doc3"])  # Predicted set
relevant_set = set(["doc1", "doc4"])           # Actually Needed set (Relevant)

# Calculate metrics
precision, recall, f_measure = calculate_metrics(retrieved_set, relevant_set)

# Print the results
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-measure: {f_measure}")
