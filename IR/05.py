import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file
df = pd.read_csv(r"C:\Users\Student\Desktop\IR\Dataset.csv")

# Combine the 'covid' and 'fever' columns to create the feature data
data = df["covid"].astype(str) + " " + df["fever"].astype(str)
X = data  # Features
y = df['flu']  # Labels (target variable)

# Splitting the data into training and test data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CountVectorizer to convert text data into a bag-of-words model
vectorizer = CountVectorizer()

# Converting the training and test data into bag-of-words format
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Initialize the Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier with the training data
classifier.fit(X_train_counts, y_train)

# Load the new dataset to test if the model is working properly
data1 = pd.read_csv(r"C:\Users\Student\Desktop\IR\Test.csv")
new_data = data1["covid"].astype(str) + " " + data1["fever"].astype(str)

# Convert the new data into bag-of-words format
new_data_counts = vectorizer.transform(new_data.astype(str))

# Predict the results for the new dataset
predictions = classifier.predict(new_data_counts)

# Output the predictions
print("Predictions for the new data: ", predictions)

# Evaluate the model performance on the test set
accuracy = accuracy_score(y_test, classifier.predict(X_test_counts))
print(f"\nAccuracy: {accuracy:.2f}")

# Print classification report
print("Classification Report: ")
print(classification_report(y_test, classifier.predict(X_test_counts)))

# Convert the predictions to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['flu_prediction'])

# Concatenate the original DataFrame with the predictions DataFrame
data1 = pd.concat([data1, predictions_df], axis=1)

# Write the DataFrame with predictions back to a new CSV file
data1.to_csv(r"C:\Users\Student\Desktop\IR\Test.csv", index=False)
