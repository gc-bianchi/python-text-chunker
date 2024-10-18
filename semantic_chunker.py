import re
import json
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
    return sentences


def compute_embeddings(sentences):
    # Use SentenceTransformer to get embeddings (free alternative)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = []
    for sentence in sentences:
        sentence_embedding = model.encode(sentence)
        embeddings.append(np.array(sentence_embedding))
    return embeddings


def find_distances(embeddings):
    distances = []
    for i in range(1, len(embeddings)):
        # Compute cosine similarity between consecutive sentence embeddings
        similarity = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        # Calculate distance as 1 - similarity
        distance = 1 - similarity
        distances.append(distance)
    return distances


def group_sentences_into_chunks(sentences, distances, threshold):
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        if i - 1 < len(distances) and distances[i - 1] > threshold:
            # If distance exceeds the threshold, start a new chunk
            chunks.append(current_chunk)
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    # Append the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def main():
    with open("./data/test_text.txt") as file:
        text = file.read()

    cleaned_sentences = preprocess_text(text)
    embeddings = compute_embeddings(cleaned_sentences)
    distances = find_distances(embeddings)

    # print(distances[:3])
    breakpoint_distance_threshold = np.percentile(distances, 95)
    # print(breakpoint_distance_threshold)

    chunks = group_sentences_into_chunks(
        cleaned_sentences, distances, breakpoint_distance_threshold
    )

    # print(chunks[:3])

    # pretty print all chunks
    for index, chunk in enumerate(chunks, start=1):
        print(f"Chunk {index}:")
        for sentence in chunk:
            print(f"  - {sentence}")
        print()

    # pretty print first 3 chunks, change integer to see more chunks printed
    # instead of printing all chunks to console
    # for index, chunk in enumerate(chunks[:3], start=1):
    #     print(f"Chunk {index}:")
    #     for sentence in chunk:
    #         print(f"  - {sentence['sentence']}")
    #     print()


main()
