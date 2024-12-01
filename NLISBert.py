import sys

from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from utils import *
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

dataset_name = sys.argv[1]
print("Dataset: ", dataset_name)

# Load dataset (or any other NLI dataset)
train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data(dataset_name)
#train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data("mnli")
#train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data("rte")
#train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data("scitail")
#train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data("anli")

# Load SBERT model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
#model = SentenceTransformer('all-mpnet-base-v2')

# Generate embeddings for premises and hypotheses
train_premise_embeddings = model.encode(train_premises, batch_size=32, show_progress_bar=True)
train_hypothesis_embeddings = model.encode(train_hypotheses, batch_size=32, show_progress_bar=True)

test_premise_embeddings = model.encode(test_premises, batch_size=32, show_progress_bar=True)
test_hypothesis_embeddings = model.encode(test_hypotheses, batch_size=32, show_progress_bar=True)

# combine embeddings for classification
train_features = combine_embeddings(train_premise_embeddings, train_hypothesis_embeddings)
test_features = combine_embeddings(test_premise_embeddings, test_hypothesis_embeddings)

'''# Train a Logistic Regression model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_features, train_labels)

# Make predictions
y_pred = classifier.predict(test_features)

# Evaluate the model by calculating metrics
accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred, average="macro", zero_division=0)
recall = recall_score(test_labels, y_pred, average="macro", zero_division=0)
f1 = f1_score(test_labels, y_pred, average="macro", zero_division=0)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")'''

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

