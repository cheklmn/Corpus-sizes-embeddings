import random

from global_params import *

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

import nltk
nltk.download('punkt')


def corpus_processing(corpus, words_count, random_articles):
    filename = "./Texts/sentences_" + SIZES_MAP[words_count] + ".txt"
    result = []
    item = iter(corpus)
    i = 0
    try:
        while i < words_count:
            if random_articles:
                current = next(item)
                while random.randint(0, 100) % 7 != 0:
                    current = next(item)
            else:
                current = next(item)
            if CROSS_SENTENCE:
                for paragraph in current["section_texts"]:
                    sentences = sent_tokenize(paragraph)
                    sent = []
                    j = 0
                    for sentence in sentences:
                        sent.append(sentence)
                        j += 1
                        if j == len(sentences) or j == 10:
                            i += make_tokens(result, ' '.join(sent))
                            sent = []
                            j = 0
            else:
                for paragraph in current["section_texts"]:
                    sentences = sent_tokenize(paragraph)
                    for sentence in sentences:
                        i += make_tokens(result, sentence)
        result = result if len(result) <= words_count else result[:words_count]
        with open(filename, "w", encoding='utf8') as file:
            file.write(str(result))
        print("Sentences file written")
    except StopIteration:
        pass
    return result


def make_tokens(result, sentence):
    words = word_tokenize(sentence)
    tokens = [word for word in words if word.isalpha()]
    result.append(tokens)
    return len(tokens)
