import re


def preprocess_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    with open("./data/test_text.txt") as file:
        text = file.read()

    cleaned_text = preprocess_text(text)

    print(cleaned_text)


main()
