from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from utils import *

# Load SimCSE model and tokenizer
model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
simcse_model = AutoModel.from_pretrained(model_name)

'''# Load SNLI dataset (or any other NLI dataset)
dataset = load_dataset('snli')
train_data = dataset['train'].shuffle(seed=42).select(range(2000))
test_data = dataset['validation'].shuffle(seed=42).select(range(400))

# Extract premises, hypotheses, and labels
train_premises = [item['premise'] for item in train_data]
train_hypotheses = [item['hypothesis'] for item in train_data]
train_labels = [item['label'] for item in train_data]  # Labels: 0 (contradiction), 1 (entailment), 2 (neutral)

test_premises = [item['premise'] for item in test_data]
test_hypotheses = [item['hypothesis'] for item in test_data]
test_labels = [item['label'] for item in test_data]'''
train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels = fetch_data("snli")
'''train_premises, train_hypotheses, train_labels = train_premises[:2000], train_hypotheses[:2000], train_labels[:2000]
test_premises, test_hypotheses, test_labels = test_premises[:500], test_hypotheses[:500], test_labels[:500]'''

# Generate embeddings for training and test sets
premise_train_embeddings = generate_embeddings(train_premises, tokenizer, simcse_model, batch_size=32)
hypothesis_train_embeddings = generate_embeddings(train_hypotheses, tokenizer, simcse_model, batch_size=32)

premise_test_embeddings = generate_embeddings(test_premises, tokenizer, simcse_model, batch_size=32)
hypothesis_test_embeddings = generate_embeddings(test_hypotheses, tokenizer, simcse_model, batch_size=32)

'''# Compute similarity (e.g., cosine similarity) between premise and hypothesis
#train_features = np.abs(premise_train_embeddings - hypothesis_train_embeddings)
#test_features = np.abs(premise_test_embeddings - hypothesis_test_embeddings)
train_similarities = util.pytorch_cos_sim(premise_train_embeddings, hypothesis_train_embeddings).diagonal().numpy().reshape(-1, 1)
test_similarities = util.pytorch_cos_sim(premise_test_embeddings, hypothesis_test_embeddings).diagonal().numpy().reshape(-1, 1)'''
# combine embeddings for classification
train_features = combine_embeddings(premise_train_embeddings, hypothesis_train_embeddings)
test_features = combine_embeddings(premise_test_embeddings, hypothesis_test_embeddings)

# Train a classifier (Logistic Regression)
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
f1 = f1_score(test_labels, y_pred, average="macro", zero_division=0)

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Save results to a text file
output_file = "results_snli.txt"

with open(output_file, "w") as file:
    file.write(f"Test Accuracy: {accuracy:.4f}\n")
    file.write(f"Test Precision: {precision:.4f}\n")
    file.write(f"Test Recall: {recall:.4f}\n")
    file.write(f"Test F1-score: {f1:.4f}\n\n")
    #file.write("Classification Report:\n")
    #file.write(report)

print(f"Results saved to {output_file}")

