import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_text(text):
    # Remove extra spaces and new lines
    text = re.sub(r"\s+", " ", text).strip()
    # Split text into sentences based on ".", "?", or "!"
    sentences = re.split(r"[.?!]\s*", text)
    # Remove any empty strings that might occur due to splitting
    sentences = [sentence.strip() for sentence in sentences if sentence]
    # Create a list of dicts with sentence text and index value
    sentence_dicts = [
        {"sentence": sentence, "index": i}
        for i, sentence in enumerate(sentences, start=1)
    ]
    return sentence_dicts


def compute_embeddings(sentences):
    # Use SentenceTransformer to get embeddings (free alternative)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = []
    for sentence_dict in sentences:
        sentence_embedding = model.encode(sentence_dict["sentence"])
        embeddings.append(np.array(sentence_embedding))
    return embeddings


def find_distances(embeddings):
    distances = []
    for i in range(1, len(embeddings)):
        # Compute cosine similarity between consecutive sentence embeddings
        similarity = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        distances.append(similarity)
    return distances


def main():
    with open("./data/test_text.txt") as file:
        text = file.read()

    cleaned_sentences = preprocess_text(text)
    embeddings = compute_embeddings(cleaned_sentences)
    distances = find_distances(embeddings)

    print(distances[:3])


main()
