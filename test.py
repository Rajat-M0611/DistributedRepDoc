from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, classification_report
import pickle
from utils import *

# Load dataset
#train, test = fetch_data("newsgroup20")
#train, test = fetch_data("rotten_tomatoes")
train, test = fetch_data("dbpedia")

# Extract data and labels
train_texts, train_labels = train.data[:2000], train.target[:2000]
test_texts, test_labels = test.data[:500], test.target[:500]
'''train_texts, train_labels = train.data, train.target
test_texts, test_labels = test.data, test.target'''

# Load SimCSE model and tokenizer
model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
simcse_model = AutoModel.from_pretrained(model_name)

train_embeddings = generate_embeddings(train_texts, tokenizer, simcse_model)
test_embeddings = generate_embeddings(test_texts, tokenizer, simcse_model)

with open("embeddings.pkl", "wb") as file:
    pickle.dump({
        "train_embeddings": train_embeddings,
        "test_embeddings": test_embeddings,
        "train_labels": train_labels,
        "test_labels": test_labels
    }, file)

print("Embeddings and labels saved successfully.")

# Train logistic regression
logistic_model = LogisticRegression(max_iter=5000)  #iterations increased to help model converge
logistic_model.fit(train_embeddings, train_labels)

# Evaluate on test data
test_predictions = logistic_model.predict(test_embeddings)

# Evaluate the model
# Evaluate the model by calculating metrics
accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(test_labels, test_predictions, average="macro", zero_division=0)
recall = recall_score(test_labels, test_predictions, average="macro", zero_division=0)
f1 = f1_score(test_labels, test_predictions, average="macro", zero_division=0)

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Classification report
#report = classification_report(test_labels, test_predictions)
#report = classification_report(test_labels, test_predictions, target_names=newsgroups_data['target_names'])
#print(report)

# Save results to a text file
output_file = "results.txt"

with open(output_file, "w") as file:
    file.write(f"Test Accuracy: {accuracy:.4f}\n")
    file.write(f"Test Precision: {precision:.4f}\n")
    file.write(f"Test Recall: {recall:.4f}\n")
    file.write(f"Test F1-score: {f1:.4f}\n\n")
    #file.write("Classification Report:\n")
    #file.write(report)

print(f"Results saved to {output_file}")

'''# Load SimCSE pretrained model and tokenizer
model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
simcse_model = AutoModel.from_pretrained(model_name)


# Function to generate embeddings
def generate_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token representation
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.squeeze(0)


# Generate embeddings for train and test sets
train_embeddings = torch.stack([generate_embedding(text, tokenizer, simcse_model) for text in train_texts])
test_embeddings = torch.stack([generate_embedding(text, tokenizer, simcse_model) for text in test_texts])

train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)


# Define a simple feedforward classifier
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Parameters
embedding_dim = train_embeddings.size(1)
num_classes = len(set(labels))

# Instantiate model
classifier = Classifier(embedding_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Create DataLoaders
train_dataset = TensorDataset(train_embeddings, train_labels)
test_dataset = TensorDataset(test_embeddings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Training loop
def train_model(model, data_loader, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for embeddings, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")


train_model(classifier, train_loader)


def evaluate_model(model, data_loader):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for embeddings, labels in data_loader:
            outputs = model(embeddings)
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Accuracy: {accuracy:.4f}")


evaluate_model(classifier, test_loader)
'''