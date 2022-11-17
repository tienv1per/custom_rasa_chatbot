import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np
stemmer = PorterStemmer()


def tokenize(sent):
    return nltk.word_tokenize(sent)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sent, all_words):
    tokenized_sent = [stem(w) for w in tokenized_sent]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sent:
            bag[idx] = 1.0
    return bag


def main():
    sent = "How long does Manchester City win the UEFA Champions League?"
    print(tokenize(sent))
    words = ["organize", "organized", "organizing"]
    for word in words:
        print(stem(word=word))
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag = bag_of_words(sentence, words)
    print(bag)


if __name__ == "__main__":
    main()
