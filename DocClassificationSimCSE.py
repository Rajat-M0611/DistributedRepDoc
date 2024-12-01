import sys
from transformers import AutoTokenizer, AutoModel
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
'''train_texts, train_labels = train.data[:2000], train.target[:2000]
test_texts, test_labels = test.data[:500], test.target[:500]'''
train_texts, train_labels = train.data, train.target
test_texts, test_labels = test.data, test.target

# Load SimCSE model and tokenizer
model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
simcse_model = AutoModel.from_pretrained(model_name)

train_embeddings = generate_embeddings(train_texts, tokenizer, simcse_model)
test_embeddings = generate_embeddings(test_texts, tokenizer, simcse_model)

# Train logistic regression
logistic_model = LogisticRegression(max_iter=5000)  # SimCSE needs more iterations to converge
logistic_model.fit(train_embeddings, train_labels)

# Evaluate on test data
test_predictions = logistic_model.predict(test_embeddings)

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

