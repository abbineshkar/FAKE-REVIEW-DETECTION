import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = [
    ("This product is amazing!", 1),      # Genuine review (1)
    ("Terrible product. Do not buy.", 0),  # Fake review (0)
    ("I love it! Best purchase ever.", 1),  # Genuine review (1)
    ("Worst product I've ever seen.", 0)   # Fake review (0)
]

# Create a DataFrame
df = pd.DataFrame(data, columns=['review', 'label'])

# Split data into text and labels
X = df['review']
y = df['label']

# Convert text data into TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Fake", "Genuine"])

print(f"Accuracy: {accuracy}")
print(report)
