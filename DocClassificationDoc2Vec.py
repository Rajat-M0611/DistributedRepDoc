import sys

from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
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

train_tagged = preprocess_and_tag(train_texts, train_labels)
test_tagged = preprocess_and_tag(test_texts, test_labels)

print("Training the model....")
# Initialize and train the Doc2Vec model
doc2vec_model = Doc2Vec(vector_size=380, window=5, min_count=2, workers=4, epochs=20)
doc2vec_model.build_vocab(train_tagged)

# Train the model
doc2vec_model.train(train_tagged, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
print("Model trained")

# Infer embeddings for train and test data
train_embeddings = infer_embeddings(doc2vec_model, train_tagged, batch_size=32)
test_embeddings = infer_embeddings(doc2vec_model, test_tagged, batch_size=32)

# Train the logistic regression model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_embeddings, train_labels)

# Make predictions
test_predictions = classifier.predict(test_embeddings)

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

#print("Classification Report:\n", classification_report(test_labels, test_predictions))
