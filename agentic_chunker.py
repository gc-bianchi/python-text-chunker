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


def generate_title(text):
    title_generation_pipeline = pipeline(
        "text2text-generation", model="TusharJoshi89/title-generator"
    )

    prompt = f"Generate a concise and descriptive title for the following text: {text}"
    generated_text = title_generation_pipeline(
        prompt, max_new_tokens=30, num_return_sequences=1
    )[0]["generated_text"].strip()
    return generated_text


def main():
    with open("./data/shorter_text.txt") as file:
        text = file.read()

    summary = summarize_text(text)
    print("Summary:")
    print(summary)

    title = generate_title(text)
    print("\nTitle:")
    print(title)


if __name__ == "__main__":
    main()
