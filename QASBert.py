import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from difflib import SequenceMatcher
import spacy
from tqdm import tqdm
from joblib import Parallel, delayed

# Load a SpaCy model for better sentence tokenization
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Load Natural Questions dataset (use a subset for demonstration)
dataset = load_dataset("natural_questions", split="train")
#dataset = dataset.select(range(100))  # Select first 100 samples for testing

# Extract questions, contexts, and answers
questions = [item['question'] for item in dataset]
contexts = [item['context'] for item in dataset]
answers = [item['answers']['text'][0] if item['answers']['text'] else '' for item in dataset]

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Evaluation function for a single sample
def evaluate_sample_batched(question, context, answer, model, batch_size=32):
    # Tokenize context into sentences using SpaCy
    doc = nlp(context)
    sentences = [sent.text for sent in doc.sents]

    # Encode the question
    question_embedding = model.encode([question], convert_to_tensor=True)

    # Encode sentences in batches
    sentence_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        sentence_embeddings.append(batch_embeddings)
    sentence_embeddings = torch.cat(sentence_embeddings, dim=0)

    # Compute cosine similarity
    cosine_scores = util.cos_sim(question_embedding, sentence_embeddings)

    # Identify the most relevant sentence
    best_sentence_index = cosine_scores.argmax()
    predicted_answer = sentences[best_sentence_index]

    # Exact match
    exact_match = int(answer.lower() in predicted_answer.lower())

    # Fuzzy matching for partial match
    fuzzy_match_score = SequenceMatcher(None, answer.lower(), predicted_answer.lower()).ratio()

    return exact_match, fuzzy_match_score

# Parallel evaluation with tqdm progress bar and batching
results = Parallel(n_jobs=-1)(
    delayed(evaluate_sample_batched)(q, c, a, model, batch_size=16)
    for q, c, a in tqdm(zip(questions, contexts, answers), total=len(answers), desc="Evaluating")
)

# Aggregate results
exact_matches = sum(result[0] for result in results)
fuzzy_match_scores = [result[1] for result in results]

# Calculate metrics
exact_match_accuracy = exact_matches / len(answers)
average_fuzzy_match = sum(fuzzy_match_scores) / len(answers)

# Optional: Fuzzy Match Threshold Metrics
threshold = 0.7
true_positives = sum(1 for score in fuzzy_match_scores if score >= threshold)
precision = true_positives / len(answers)
recall = true_positives / len(answers)
f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

# Print results
print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
print(f"Average Fuzzy Match Score: {average_fuzzy_match:.4f}")
print(f"Precision (Fuzzy Match ≥ {threshold}): {precision:.4f}")
print(f"Recall (Fuzzy Match ≥ {threshold}): {recall:.4f}")
print(f"F1-Score (Fuzzy Match ≥ {threshold}): {f1:.4f}")
