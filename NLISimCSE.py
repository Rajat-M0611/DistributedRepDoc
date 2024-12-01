import sys
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch.nn as nn
import torch.optim as optim
from utils import *
import pickle

# Load SimCSE model and tokenizer
model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
simcse_model = AutoModel.from_pretrained(model_name)

dataset_name = sys.argv[1]
print("Dataset: ", dataset_name)

train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data(dataset_name)
#train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data("snli")
#train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data("mnli")
#train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data("rte")
#train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data("scitail")
#train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data("anli")
'''train_premises, train_hypotheses, train_labels = train_premises[:200], train_hypotheses[:200], train_labels[:200]
test_premises, test_hypotheses, test_labels = test_premises[:50], test_hypotheses[:50], test_labels[:50]'''

# Generate embeddings for training and test sets
premise_train_embeddings = generate_embeddings(train_premises, tokenizer, simcse_model, batch_size=32)
hypothesis_train_embeddings = generate_embeddings(train_hypotheses, tokenizer, simcse_model, batch_size=32)

premise_test_embeddings = generate_embeddings(test_premises, tokenizer, simcse_model, batch_size=32)
hypothesis_test_embeddings = generate_embeddings(test_hypotheses, tokenizer, simcse_model, batch_size=32)

with open("embeddings1.pkl", "wb") as file:
    pickle.dump({
        "premise_train_embeddings": premise_train_embeddings,
        "premise_test_embeddings": premise_test_embeddings,
        "hypothesis_train_embeddings": hypothesis_train_embeddings,
        "hypothesis_test_embeddings": hypothesis_test_embeddings,
        "train_labels": train_labels,
        "test_labels": test_labels
    }, file)

print("Embeddings and labels saved successfully.")

# combine embeddings for classification
train_features = combine_embeddings(premise_train_embeddings, hypothesis_train_embeddings)
test_features = combine_embeddings(premise_test_embeddings, hypothesis_test_embeddings)

'''# Train a classifier (Logistic Regression)
classifier = LogisticRegression(max_iter=5000)  # SimCSE needs more iterations to converge
#classifier.fit(train_features, train_labels)
#classifier.fit(train_similarities, train_labels)
classifier.fit(train_features, train_labels)

# Predict on test set
#y_pred = classifier.predict(test_features)
#y_pred = classifier.predict(test_similarities)
y_pred = classifier.predict(test_features)

# Evaluate the model by calculating metrics
accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred, average="macro", zero_division=0)
recall = recall_score(test_labels, y_pred, average="macro", zero_division=0)
f1 = f1_score(test_labels, y_pred, average="macro", zero_division=0)'''

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
# print(train_labels.min(), train_labels.max())  # Should be [0, num_classes-1]

# Parameters
embedding_dim = train_features.shape[1]
num_classes = len(set(train_labels.numpy()))  # Convert tensor to numpy for unique count

# print("num_classes: ", num_classes)

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

# Save results to a text file
output_file = "results_anli.txt"

with open(output_file, "w") as file:
    file.write(f"Test Accuracy: {accuracy:.4f}\n")
    file.write(f"Test Precision: {precision:.4f}\n")
    file.write(f"Test Recall: {recall:.4f}\n")
    file.write(f"Test F1-score: {f1:.4f}\n\n")
    #file.write("Classification Report:\n")
    #file.write(report)

print(f"Results saved to {output_file}")
