import sys

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, classification_report
from utils import *

dataset_name = sys.argv[1]
print("Dataset: ", dataset_name)

# load data from dataset
#train, test = fetch_data("newsgroup20")
#train, test = fetch_data("rotten_tomatoes")
train, test = fetch_data(dataset_name)

# Extract data and labels
train_texts, train_labels = train.data, train.target
test_texts, test_labels = test.data, test.target

# Load pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for training and testing data
train_embeddings = model.encode(train_texts, show_progress_bar=True)
test_embeddings = model.encode(test_texts, show_progress_bar=True)

# Train the logistic regression model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_embeddings, train_labels)

# Make predictions
test_predictions = classifier.predict(test_embeddings)

# Evaluate the model
# Evaluate the model by calculating metrics
accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(test_labels, test_predictions, average="micro", zero_division=0)
recall = recall_score(test_labels, test_predictions, average="micro", zero_division=0)
f1 = f1_score(test_labels, test_predictions, average="micro", zero_division=0)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

