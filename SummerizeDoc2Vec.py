import sys
import torch
from datasets import load_dataset
import nltk
from rouge_score import rouge_scorer
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Download punkt tokenizer for sentence splitting
nltk.download('punkt')

# Step 1: Load CNN/Daily Mail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Use a subset of the test dataset for demonstration
test_data = dataset['test']#.select(range(100))  # Adjust the range for larger subsets

# Extract articles and summaries
test_articles = [entry['article'] for entry in test_data]
test_summaries = [entry['highlights'] for entry in test_data]

# Step 2: Prepare Data for Doc2Vec
def prepare_data_for_doc2vec(documents):
    tagged_documents = []
    for i, document in enumerate(documents):
        sentences = nltk.sent_tokenize(document)
        for j, sentence in enumerate(sentences):
            tagged_documents.append(TaggedDocument(words=sentence.split(), tags=[f"{i}_{j}"]))
    return tagged_documents

# Step 3: Train or Load Doc2Vec Model
def train_doc2vec_model(tagged_documents, vector_size=100, window=5, min_count=1, epochs=20):
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=4)
    model.build_vocab(tagged_documents)
    model.train(tagged_documents, total_examples=model.corpus_count, epochs=epochs)
    return model

# Prepare data for Doc2Vec
tagged_documents = prepare_data_for_doc2vec(test_articles)

# Train a Doc2Vec model on the dataset (you can also load a pre-trained model)
doc2vec_model = train_doc2vec_model(tagged_documents)

# Step 4: Generate Sentence Embeddings
def generate_doc2vec_embeddings(documents, model):
    embeddings_batch = []
    for document in documents:
        sentences = nltk.sent_tokenize(document)
        sentence_embeddings = []
        for sentence in sentences:
            # Use Doc2Vec to infer a vector for each sentence
            embedding = model.infer_vector(sentence.split())
            sentence_embeddings.append(embedding)
        # Average the sentence embeddings to get a document embedding
        doc_embedding = np.mean(sentence_embeddings, axis=0)
        embeddings_batch.append((sentences, sentence_embeddings, doc_embedding))
    return embeddings_batch

# Generate embeddings for test articles using Doc2Vec
test_embeddings = generate_doc2vec_embeddings(test_articles, doc2vec_model)

# Step 5: Score Sentences and Extract Summary
def extract_summary(sentence_embeddings, top_k=3):
    summary = []
    for sentences, sentence_embeddings, doc_embedding in sentence_embeddings:
        if len(sentences) <= top_k:  # Handle cases with fewer sentences than `top_k`
            summary.append(' '.join(sentences))
            continue
        # Compute cosine similarity between sentence embeddings and the document embedding
        similarities = np.dot(sentence_embeddings, doc_embedding) / (np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(doc_embedding))
        similar_sentences_indices = np.argsort(similarities)[-top_k:]
        summary_sentences = [sentences[i] for i in similar_sentences_indices]
        summary.append(' '.join(summary_sentences))
    return summary

# Step 6: Parallel Summary Extraction
def parallel_extract_summaries(sentence_embeddings, top_k=3, num_threads=4):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda x: extract_summary([x], top_k=top_k)[0], sentence_embeddings))
    return results

# Get summaries for the test set
test_summaries_pred = parallel_extract_summaries(test_embeddings, top_k=int(sys.argv[1]))

# Step 7: ROUGE Score Calculation
def calculate_rouge(predicted_summaries, true_summaries):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_results = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for pred, true in zip(predicted_summaries, true_summaries):
        scores = scorer.score(true, pred)
        for key in rouge_results:
            rouge_results[key].append(scores[key].fmeasure)

    avg_rouge = {key: np.mean(value) for key, value in rouge_results.items()}
    return avg_rouge

# Step 8: Evaluate the Results
rouge_scores = calculate_rouge(test_summaries_pred, test_summaries)

# Step 9: Print Results
print(f"ROUGE Scores:")
print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
