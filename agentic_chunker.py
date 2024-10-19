import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def preprocess_text(text):
    # Remove extra spaces and new lines
    text = re.sub(r"\s+", " ", text).strip()
    # Split text into sentences based on ".", "?", or "!"
    sentences = re.split(r"[.?!]\s*", text)
    # Remove any empty strings that might occur due to splitting
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences


def summarize_text(text, max_summary_length=100):
    summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarization_pipeline(
        text, max_length=max_summary_length, min_length=30, do_sample=False
    )[0]["summary_text"]
    return summary


def generate_title(text):
    title_generation_pipeline = pipeline(
        "text2text-generation", model="TusharJoshi89/title-generator"
    )
    prompt = f"Generate a concise and descriptive title for the following text: {text}"
    generated_text = title_generation_pipeline(
        prompt, max_new_tokens=30, num_return_sequences=1
    )[0]["generated_text"].strip()
    return generated_text


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
    with open("./data/shorter_text.txt") as file:
        text = file.read()

    cleaned_sentences = preprocess_text(text)
    embeddings = compute_embeddings(cleaned_sentences)
    distances = find_distances(embeddings)

    breakpoint_distance_threshold = np.percentile(distances, 95)

    chunks = group_sentences_into_chunks(
        cleaned_sentences, distances, breakpoint_distance_threshold
    )

    summary = summarize_text(text)
    print("Summary:")
    print(summary)

    title = generate_title(text)
    print("\nTitle:")
    print(title)

    print("\nChunks:")
    for index, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {index}:")
        for sentence in chunk:
            print(f"  - {sentence}")


if __name__ == "__main__":
    main()
