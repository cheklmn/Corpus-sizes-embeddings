from corpus_sizes.global_params import *
from corpus_sizes.corpus_processing import corpus_processing
import gensim.downloader as api
from gensim.models import Word2Vec
from web.evaluate import evaluate_on_all
from web.embedding import Embedding
from web.datasets.similarity import fetch_MEN, fetch_MTurk, fetch_RG65, fetch_RW, fetch_SimLex999, fetch_TR9856, fetch_WS353
from collections import Counter
from operator import itemgetter


def get_dataset(dataset_name):
    if dataset_name == "WS353":
        return fetch_WS353("similarity")
    elif dataset_name == "MEN":
        return fetch_MEN("all")
    elif dataset_name == "SimLex-999":
        return fetch_SimLex999()
    elif dataset_name == "MTurk":
        return fetch_MTurk()
    elif dataset_name == "WS353":
        return fetch_WS353('all')
    elif dataset_name == "RG-65":
        return fetch_RG65()
    elif dataset_name == "RW":
        return fetch_RW()
    elif dataset_name == "TR9856":
        return fetch_TR9856()
    else:
        raise Exception("{}: dataset not supported".format(dataset_name))


def single_word_bins(tks, vocab):
    c = Counter(word for sentence in tks for word in set(sentence))

    word_freqs = [(word, c[word]) for word in vocab]
    word_freqs.sort(key = itemgetter(1, 0))

    words = [word for (word, freq) in word_freqs]
    data_len = len(word_freqs)
    part_length = data_len // 3
    high = words[2 * part_length:]
    high_bounds = (c[high[0]], c[high[-1]])
    mid = words[part_length:2 * part_length]
    mid_bounds = (c[mid[0]], c[mid[-1]])
    low = words[:part_length]
    low_bounds = (c[low[0]], c[low[-1]])
    return high, mid, low, high_bounds, mid_bounds, low_bounds


corpus = api.load("wiki-english-20171001")

for word_count in SIZES:
    tokens = corpus_processing(corpus, word_count, RANDOMIZE_ARTICLES)
    for word_window in WORD_WINDOWS:
        model_filename = "./Models/w2v_" + SIZES_MAP[word_count] + "_" + str(word_window)
        if not USE_CACHED:
            model = Word2Vec(tokens, size=300, window=word_window, workers=4)
            model.wv.save_word2vec_format(model_filename, binary=False)
            print("Model saved at " + model_filename)
        embedding_model = Embedding.from_word2vec(model_filename, binary=False)
        results = evaluate_on_all(embedding_model)
        csv_filename = "./Results/w2v" + SIZES_MAP[word_count] + "_" + str(word_window) + ".csv"
        results.to_csv(csv_filename)
        print("Results saved at " + csv_filename)

