from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset
import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

nltk.download('movie_reviews')


class CustomDataset:
    def __init__(self, data, target):
        self.data = data  # Store the text data
        self.target = target  # Store the labels


# Preprocessing and tagging with batching and tqdm
def preprocess_and_tag(texts, labels, batch_size=32):
    tagged_documents = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Preprocessing"):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        for j, (doc, label) in enumerate(zip(batch_texts, batch_labels)):
            tokens = simple_preprocess(doc)  # Tokenize the document
            tagged_documents.append(TaggedDocument(tokens, [f"TAG_{i + j}"]))  # Add tags
    return tagged_documents


# Preprocessing and tagging for NLI Doc2Vec
def preprocess_and_tag_nli(sentences, tag_prefix):
    tagged_docs = []
    for i, sentence in enumerate(sentences):
        tokens = simple_preprocess(sentence)  # Tokenize
        tagged_docs.append(TaggedDocument(tokens, [f"{tag_prefix}_{i}"]))
    return tagged_docs


# Generate embeddings for new documents for Doc2Vec
def infer_embeddings(model, tagged_docs, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(tagged_docs), batch_size), desc="Generating Embeddings"):
        batch_docs = tagged_docs[i:i + batch_size]
        batch_embeddings = [
            model.infer_vector(doc.words) for doc in batch_docs
        ]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)


# Generate embeddings for new documents for SimCSE
def generate_embeddings(texts, tokenizer, model, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


# combine embeddings for NLI Doc2Vec
def combine_embeddings(embeddings1, embeddings2):
    return np.concatenate([
        embeddings1,
        embeddings2,
        np.abs(embeddings1 - embeddings2),  # Difference
        embeddings1 * embeddings2  # Element-wise product
    ], axis=1)


# Define a simple feedforward classifier
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# Evaluation loop
def evaluate_model(model, data_loader, device):
    model.eval()
    '''total_correct = 0
    total_samples = 0'''
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            predictions = torch.argmax(outputs, dim=1)
            '''total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)'''
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    '''accuracy = total_correct / total_samples
    print(f"Accuracy: {accuracy:.4f}")'''
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

    '''print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")'''

    return accuracy, precision, recall, f1


# Training loop
def train_and_evaluate(model, data_loader, criterion, optimizer, device, classifier, test_loader, num_epochs=50):
    model.train()
    # Store the highest accuracy during training
    max_accuracy, max_precision, max_recall, max_f1 = 0.0, 0.0, 0.0, 0.0  # Initialize the max accuracy
    for epoch in range(num_epochs):
        total_loss = 0
        for embeddings, labels in data_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)

            # Debugging shapes
            # print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")

        #evaluate_model(classifier, test_loader, device)
        # Evaluate after each epoch
        accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)

        # Update max accuracy
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_precision = precision
            max_recall = recall
            max_f1 = f1
            #print(f"New highest accuracy: {max_accuracy:.4f}")

    return max_accuracy, max_precision, max_recall, max_f1


# fetch train and test data from a given dataset
def fetch_data(dataset_name):
    if dataset_name == "newsgroup20":
        # Load the 20 Newsgroups dataset
        train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

        return train, test

    elif dataset_name == "rotten_tomatoes":
        # Load Rotten Tomatoes (movie_reviews) dataset
        documents = [(list(movie_reviews.words(fileid)), category)
                     for category in movie_reviews.categories()
                     for fileid in movie_reviews.fileids(category)]

        # Split into texts and labels
        texts = [" ".join(doc) for doc, _ in documents]  # Convert tokens back to text
        labels = [1 if label == 'pos' else 0 for _, label in documents]  # Positive = 1, Negative = 0

        # Split into training and testing sets
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        train = CustomDataset(data=train_texts, target=train_labels)
        test = CustomDataset(data=test_texts, target=test_labels)

        return train, test

    elif dataset_name == "dbpedia":
        # Load the DBpedia dataset
        dataset = load_dataset("dbpedia_14")

        # Randomly shuffle and select 16,000 samples from training and 2,000 samples from test datasets
        # ratio of train/test = 8 same as in original dataset (560k/70k)
        train_subset = dataset['train'].shuffle(seed=42).select(range(16000))
        test_subset = dataset['test'].shuffle(seed=42).select(range(2000))

        # Extract data and labels
        train_texts = train_subset['content']
        train_labels = train_subset['label']
        test_texts = test_subset['content']
        test_labels = test_subset['label']

        train = CustomDataset(data=train_texts, target=train_labels)
        test = CustomDataset(data=test_texts, target=test_labels)

        return train, test

    elif dataset_name == "snli":
        # load slni dataset
        dataset = load_dataset('snli')
        train_data = dataset['train'].shuffle(seed=42).select(range(10000))
        test_data = dataset['validation'].shuffle(seed=42).select(range(2000))

        # Extract premises, hypotheses, and labels
        train_premises = [item['premise'] for item in train_data]
        train_hypotheses = [item['hypothesis'] for item in train_data]
        train_labels = [item['label'] for item in train_data]  # Labels: 0 (contradiction), 1 (entailment), 2 (neutral)

        test_premises = [item['premise'] for item in test_data]
        test_hypotheses = [item['hypothesis'] for item in test_data]
        test_labels = [item['label'] for item in test_data]

        return train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels

    elif dataset_name == "mnli":
        # load mlni dataset
        dataset = load_dataset("multi_nli")
        train_data = dataset['train'].shuffle(seed=42).select(range(10000))  # Select 10,000 samples for training
        test_data = dataset['validation_matched'].shuffle(seed=42).select(range(2000))  # Select 2,000 samples for testing

        # Extract premises, hypotheses, and labels
        train_premises = [item['premise'] for item in train_data]
        train_hypotheses = [item['hypothesis'] for item in train_data]
        train_labels = [item['label'] for item in train_data]  # Labels: 0 (contradiction), 1 (entailment), 2 (neutral)

        test_premises = [item['premise'] for item in test_data]
        test_hypotheses = [item['hypothesis'] for item in test_data]
        test_labels = [item['label'] for item in test_data]

        return train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels

    elif dataset_name == "rte":
        # Load RTE dataset
        dataset = load_dataset("glue", "rte")
        train_data = dataset['train']  # Use the training split
        test_data = dataset['validation']  # Use the validation split for testing

        # Extract premises, hypotheses, and labels
        train_premises = train_data['sentence1']
        train_hypotheses = train_data['sentence2']
        train_labels = train_data['label']  # Labels: 0 (entailment), 1 (not entailment)

        test_premises = test_data['sentence1']
        test_hypotheses = test_data['sentence2']
        test_labels = test_data['label']

        return train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels

    elif dataset_name == "anli":
        # Load ANLI dataset
        dataset = load_dataset("anli")
        train_data = dataset['train_r1']  # Use round 1 for training; adjust to 'train_r2' or 'train_r3' for other rounds
        test_data = dataset['dev_r1']  # Use round 1 for testing; adjust to 'dev_r2' or 'dev_r3' for other rounds

        # Extract premises, hypotheses, and labels
        train_premises = train_data['premise']
        train_hypotheses = train_data['hypothesis']
        train_labels = train_data['label']  # Labels: 0 (contradiction), 1 (entailment), 2 (neutral)

        test_premises = test_data['premise']
        test_hypotheses = test_data['hypothesis']
        test_labels = test_data['label']

        return train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels

    elif dataset_name == "scitail":
        # Load SciTail dataset
        dataset = load_dataset("scitail", 'dgem_format')
        train_data = dataset['train']
        test_data = dataset['test']

        # Extract premises, hypotheses, and labels
        train_premises = train_data['premise']
        train_hypotheses = train_data['hypothesis']
        train_labels = train_data['label']  # Labels: 0 (contradiction), 1 (entailment)

        test_premises = test_data['premise']
        test_hypotheses = test_data['hypothesis']
        test_labels = test_data['label']

        return train_premises, train_hypotheses, train_labels, test_premises, test_hypotheses, test_labels

    elif dataset_name == "swag":
        # Load SWAG dataset
        dataset = load_dataset("swag")
        train_data = dataset['train']
        test_data = dataset['test']

        # Extract sentences and options for training
        train_sentences = train_data['sentence']
        train_options = train_data['ending']
        train_labels = train_data['label']  # Labels: 0 (option A), 1 (option B), 2 (option C), 3 (option D)

        test_sentences = test_data['sentence']
        test_options = test_data['ending']
        test_labels = test_data['label']

        return train_sentences, train_options, train_labels, test_sentences, test_options, test_labels

    else:
        print("Unhandled dataset: ", dataset_name)
        return None

