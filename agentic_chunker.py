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

    title_generation_pipeline = pipeline(
        "text-generation", model="EleutherAI/gpt-neo-125M"
    )

    prompt = f"Create a concise, 3-5 word title for the following summary:\nSummary: {summary}\nTitle:"
    generated_text = title_generation_pipeline(
        prompt, max_new_tokens=5, num_return_sequences=1
    )[0]["generated_text"]
    # Extract title by removing the prompt part
    title = generated_text.split("Title:")[-1].strip()
    # Limit title to 5 words or fewer
    title = " ".join(title.split()[:5])
    return title


def main():
    # test text
    text = (
        "The rise of artificial intelligence (AI) has been one of the most significant technological developments "
        "of the 21st century. AI has been applied across industries, revolutionizing healthcare, finance, transportation, "
        "and many other fields. However, with great power comes great responsibility, and ethical concerns have arisen "
        "regarding the potential misuse of AI, bias in algorithms, and the impact on jobs. To ensure that AI is developed "
        "and applied in a beneficial way, it is crucial to establish guidelines and regulations that foster transparency, "
        "fairness, and accountability."
    )

    summary = summarize_text(text)
    print("Summary:")
    print(summary)

    title = generate_title(summary)
    print("\nTitle:")
    print(title)


if __name__ == "__main__":
    main()
