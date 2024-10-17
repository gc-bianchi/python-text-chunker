import re


def preprocess_text(text):
    # Remove extra spaces and new lines
    text = re.sub(r"\s+", " ", text).strip()
    # Split text into sentences based on ".", "?", or "!"
    sentences = re.split(r"[.?!]\s*", text)
    # Remove any empty strings that might occur due to splitting with 'if sentence'
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences


def main():
    with open("./data/test_text.txt") as file:
        text = file.read()

    cleaned_sentences = preprocess_text(text)

    print(cleaned_sentences)


main()
