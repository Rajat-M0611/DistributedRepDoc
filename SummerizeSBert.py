import sys
import torch
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import nltk
from rouge_score import rouge_scorer
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Download punkt tokenizer for sentence splitting
nltk.download('punkt')

# Step 1: Load CNN/Daily Mail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Use a subset of the test dataset for demonstration
test_data = dataset['test']#.select(range(100))  # Adjust the range for larger subsets

# Extract articles and summaries
test_articles = [entry['article'] for entry in test_data]
test_summaries = [entry['highlights'] for entry in test_data]

# Step 2: Load SBERT Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Generate Sentence Embeddings with Batching
def generate_sentence_embeddings(documents, batch_size=32):
    all_embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        embeddings_batch = []
        for document in batch:
            sentences = nltk.sent_tokenize(document)
            embeddings = model.encode(sentences, convert_to_tensor=True, batch_size=32)
            doc_embedding = embeddings.mean(dim=0, keepdim=True)
            embeddings_batch.append((sentences, embeddings, doc_embedding))
        all_embeddings.extend(embeddings_batch)
    return all_embeddings

# Generate embeddings for test articles
test_embeddings = generate_sentence_embeddings(test_articles)

# Step 4: Score Sentences and Extract Summary
def extract_summary(sentence_embeddings, top_k=3):
    summary = []
    for sentences, embeddings, doc_embedding in sentence_embeddings:
        if len(sentences) <= top_k:  # Handle cases with fewer sentences than `top_k`
            summary.append(' '.join(sentences))
            continue
        similarities = util.pytorch_cos_sim(embeddings, doc_embedding)
        similar_sentences = torch.topk(similarities.squeeze(), top_k).indices
        summary_sentences = [sentences[i] for i in similar_sentences]
        summary.append(' '.join(summary_sentences))
    return summary

# Step 5: Parallel Summary Extraction
def parallel_extract_summaries(sentence_embeddings, top_k=3, num_threads=4):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda x: extract_summary([x], top_k=top_k)[0], sentence_embeddings))
    return results

# Get summaries for the test set
test_summaries_pred = parallel_extract_summaries(test_embeddings, top_k=int(sys.argv[1]))

# Step 6: ROUGE Score Calculation
def calculate_rouge(predicted_summaries, true_summaries):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_results = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for pred, true in zip(predicted_summaries, true_summaries):
        scores = scorer.score(true, pred)
        for key in rouge_results:
            rouge_results[key].append(scores[key].fmeasure)

    avg_rouge = {key: np.mean(value) for key, value in rouge_results.items()}
    return avg_rouge

# Step 7: Evaluate the Results
rouge_scores = calculate_rouge(test_summaries_pred, test_summaries)

# Step 8: Print Results
print(f"ROUGE Scores:")
print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
