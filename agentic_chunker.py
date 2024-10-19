import os
from transformers import pipeline

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def summarize_text(text, max_summary_length=100):

    summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

    summary = summarization_pipeline(
        text, max_length=max_summary_length, min_length=30, do_sample=False
    )[0]["summary_text"]
    return summary


def generate_title(summary):

    # Use the title generator model from Hugging Face with text2text-generation pipeline
    title_generation_pipeline = pipeline(
        "text2text-generation", model="TusharJoshi89/title-generator"
    )

    prompt = f"Provide a short, descriptive, and broad title that includes all key contributors (Galileo, Newton, Einstein) and their impact in the following summary: {summary}"
    generated_text = title_generation_pipeline(
        prompt, max_new_tokens=25, num_return_sequences=1
    )[0]["generated_text"].strip()
    return generated_text


def main():
    with open("./data/shorter_text.txt") as file:
        text = file.read()

    summary = summarize_text(text)
    print("Summary:")
    print(summary)

    title = generate_title(summary)
    print("\nTitle:")
    print(title)


if __name__ == "__main__":
    main()
