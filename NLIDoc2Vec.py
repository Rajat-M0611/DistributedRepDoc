import sys

from datasets import load_dataset
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

dataset_name = sys.argv[1]
print("Dataset: ", dataset_name)

# Load dataset
#train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data("snli")
#train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data("mnli")
#train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data("rte")
#train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data("scitail")
train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data(dataset_name)

# Create tagged documents for premises and hypotheses
train_premise_docs = preprocess_and_tag_nli(train_premises, "TRAIN_PREMISE")
train_hypothesis_docs = preprocess_and_tag_nli(train_hypotheses, "TRAIN_HYPOTHESIS")

test_premise_docs = preprocess_and_tag_nli(test_premises, "TEST_PREMISE")
test_hypothesis_docs = preprocess_and_tag_nli(test_hypotheses, "TEST_HYPOTHESIS")

# Combine all tagged documents for training
all_train_docs = train_premise_docs + train_hypothesis_docs

# Initialize and train the Doc2Vec model
doc2vec_model = Doc2Vec(vector_size=380, window=5, min_count=2, workers=4, epochs=20)
doc2vec_model.build_vocab(all_train_docs)
doc2vec_model.train(all_train_docs, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

# Infer embeddings for train and test data
train_premise_embeddings = infer_embeddings(doc2vec_model, train_premise_docs, batch_size=32)
train_hypothesis_embeddings = infer_embeddings(doc2vec_model, train_hypothesis_docs, batch_size=32)

test_premise_embeddings = infer_embeddings(doc2vec_model, test_premise_docs, batch_size=32)
test_hypothesis_embeddings = infer_embeddings(doc2vec_model, test_hypothesis_docs, batch_size=32)

# combine embeddings for classification
train_features = combine_embeddings(train_premise_embeddings, train_hypothesis_embeddings)
test_features = combine_embeddings(test_premise_embeddings, test_hypothesis_embeddings)

'''# Compute cosine similarities
train_similarities = np.array([
    cosine_similarity(prem.reshape(1, -1), hyp.reshape(1, -1))[0, 0]
    for prem, hyp in zip(train_premise_embeddings, train_hypothesis_embeddings)
])

test_similarities = np.array([
    cosine_similarity(prem.reshape(1, -1), hyp.reshape(1, -1))[0, 0]
    for prem, hyp in zip(test_premise_embeddings, test_hypothesis_embeddings)
])'''

# Train a Logistic Regression model
'''classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_features, train_labels)

# Make predictions
test_predictions = classifier.predict(test_features)'''
'''# Reshape similarity scores for compatibility with sklearn models
train_similarities = train_similarities.reshape(-1, 1)
test_similarities = test_similarities.reshape(-1, 1)

# Train a logistic regression model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_similarities, train_labels)

# Predict labels for test data
test_predictions = classifier.predict(test_similarities)

# Evaluate the model
accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(test_labels, test_predictions, average="macro", zero_division=0)
recall = recall_score(test_labels, test_predictions, average="macro", zero_division=0)
f1 = f1_score(test_labels, test_predictions, average="macro", zero_division=0)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
'''

# converts string labels to numeric for datasets like Scitail
if isinstance(train_labels[0], str):
    label_map = {"entails": 0, "neutral": 1}
    train_labels = [label_map[label] for label in train_labels]
    test_labels = [label_map[label] for label in test_labels]

# Convert features and labels to PyTorch tensors
train_features = torch.tensor(train_features, dtype=torch.float32)
test_features = torch.tensor(test_features, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Remove samples with label = -1
train_mask = train_labels != -1
train_features = train_features[train_mask]
train_labels = train_labels[train_mask]

test_mask = test_labels != -1
test_features = test_features[test_mask]
test_labels = test_labels[test_mask]

# Check ranges
#print(train_labels.min(), train_labels.max())  # Should be [0, num_classes-1]

# Parameters
embedding_dim = train_features.shape[1]
num_classes = len(set(train_labels.numpy()))  # Convert tensor to numpy for unique count

#print("num_classes: ", num_classes)

# Instantiate model
classifier = Classifier(embedding_dim, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Create DataLoaders
train_dataset = TensorDataset(train_features, train_labels)
test_dataset = TensorDataset(test_features, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)

# train and evaluate model
accuracy, precision, recall, f1 = train_and_evaluate(classifier, train_loader, criterion, optimizer, device,
                                                     classifier, test_loader, num_epochs=50)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

