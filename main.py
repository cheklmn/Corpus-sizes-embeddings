from corpus_sizes.global_params import *
from corpus_sizes.corpus_processing import corpus_processing

import gensim.downloader as api
from gensim.models import Word2Vec

from web.evaluate import evaluate_on_all
from web.embedding import Embedding
from web.datasets.similarity import fetch_MEN, fetch_MTurk, fetch_RG65, fetch_RW, fetch_SimLex999, fetch_TR9856, fetch_WS353
from web.evaluate import evaluate_similarity

from collections import Counter
from operator import itemgetter

import numpy as np
import pandas as pd
import pickle
import os
import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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
    elif dataset_name == "RG65":
        return fetch_RG65()
    elif dataset_name == "RW":
        return fetch_RW()
    elif dataset_name == "TR9856":
        return fetch_TR9856()
    else:
        raise Exception("{}: dataset not supported".format(dataset_name))


def single_word_bins(sentences, vocabulary):
    c = Counter(word for sentence in sentences for word in set(sentence))

    word_freqs = [(word, c[word]) for word in vocabulary]
    word_freqs.sort(key = itemgetter(1, 0))

    words = [word for (word, freq) in word_freqs]
    data_len = len(word_freqs)
    split = data_len // 3

    high = words[2 * split:]
    high_bounds = (c[high[0]], c[high[-1]])
    mid = words[split:2 * split]
    mid_bounds = (c[mid[0]], c[mid[-1]])
    low = words[:split]
    low_bounds = (c[low[0]], c[low[-1]])

    return high, mid, low, high_bounds, mid_bounds, low_bounds


def pair_bins(low, mid, high, dataset):
    pair_similarity ={}
    low_pairs = []
    mid_pairs = []
    high_pairs = []
    mixed_pairs = []

    low_similarity = []
    mid_similarity = []
    high_similarity = []
    mixed_similarity = []

    for pair, similarity in zip(dataset.X, dataset.y):
        if pair[0] in low and pair[1] in low:
            low_pairs.append(pair)
            low_similarity.append(similarity)
            continue
        if pair[0] in mid and pair[1] in mid:
            mid_pairs.append(pair)
            mid_similarity.append(similarity)
            continue
        if pair[0] in high and pair[1] in high:
            high_pairs.append(pair)
            high_similarity.append(similarity)
            continue
        mixed_pairs.append(pair)
        mixed_similarity.append(similarity)

    pair_similarity['low'] = (low_pairs, low_similarity)
    pair_similarity['mid'] = (mid_pairs, mid_similarity)
    pair_similarity['high'] = (high_pairs, high_similarity)
    pair_similarity['mixed'] = (mixed_pairs, mixed_similarity)

    return pair_similarity


def save_results(gen_similarity, low_similarity, mid_similarity, high_similarity, mixed_similarity,
                 low_count, mid_count, high_count, mixed_count, low_bounds, mid_bounds, high_bounds,
                 word_window, dataset_name, corpus_size):
    method = "SG" if SKIP_GRAMS else "CBOW"
    results_file = './Results/Results.pickle'

    if os.path.isfile(results_file):
        df = pd.read_pickle(results_file)
    else:
        df = pd.DataFrame()
    df = df.append(pd.DataFrame({
        "Method": method,
        "Time": datetime.datetime.now(),
        "Window": word_window,
        "Word count": corpus_size,
        "Dataset": dataset_name,
        "Low bin lower bound": low_bounds[0],
        "Low bin upper bound": low_bounds[1],
        "Mid bin lower bound": mid_bounds[0],
        "Mid bin upper bound": mid_bounds[1],
        "High bin lower bound": high_bounds[0],
        "High bin upper bound": high_bounds[1],
        "Low bin score": low_similarity,
        "Low bin pair count": low_count,
        "Middle bin score": mid_similarity,
        "Middle bin pair count": mid_count,
        "High bin score": high_similarity,
        "High bin pair count": high_count,
        "Mixed bin score": mixed_similarity,
        "Mixed bin pair count": mixed_count,
        "General score": gen_similarity,
    }, index=[0]), ignore_index=True)

    df.to_pickle(results_file)
    print(df)


corpus = api.load("wiki-english-20171001")

for word_count in SIZES:
    sents = corpus_processing(corpus, word_count, RANDOMIZE_ARTICLES)

    for word_window in WORD_WINDOWS:
        model_filename = "./Models/w2v_" + SIZES_MAP[word_count] + "_" + str(word_window)

        if not USE_CACHED:
            skip_grams = 1 if SKIP_GRAMS else 0
            model = Word2Vec(sents, size=300, window = word_window, workers=4, sg=skip_grams)
            model.wv.save_word2vec_format(model_filename, binary=False)
            print("Model saved at " + model_filename)

        embedding_model = Embedding.from_word2vec(model_filename, binary=False)

        results = evaluate_on_all(embedding_model)
        csv_filename = "./Results/w2v" + SIZES_MAP[word_count] + "_" + str(word_window) + ".csv"
        results.to_csv(csv_filename)
        print("Results saved at " + csv_filename)

        for dset in USE_DATASETS:
            dataset = get_dataset(dset)
            vocab = set()
            for pair in dataset.X:
                vocab.add(pair[0])
                vocab.add(pair[1])

            high, mid, low, high_bounds, mid_bounds, low_bounds = single_word_bins(sents, vocab)
            pair_similarity = pair_bins(low, mid, high, dataset)

            model_general_similarity = evaluate_similarity(embedding_model, dataset.X, dataset.y)
            model_low_similarity = evaluate_similarity(embedding_model,
                                                       np.asarray(pair_similarity['low'][0]),
                                                       np.asarray(pair_similarity['low'][1])
                                                       )
            model_mid_similarity = evaluate_similarity(embedding_model,
                                                       np.asarray(pair_similarity['mid'][0]),
                                                       np.asarray(pair_similarity['mid'][1])
                                                       )
            model_high_similarity = evaluate_similarity(embedding_model,
                                                        np.asarray(pair_similarity['high'][0]),
                                                        np.asarray(pair_similarity['high'][1])
                                                        )
            model_mixed_similarity = evaluate_similarity(embedding_model,
                                                         np.asarray(pair_similarity['mixed'][0]),
                                                         np.asarray(pair_similarity['mixed'][1])
                                                         )
            save_results(model_general_similarity, model_low_similarity, model_mid_similarity, model_high_similarity,
                         model_mixed_similarity, len(pair_similarity['low'][0]), len(pair_similarity['mid'][0]),
                         len(pair_similarity['high'][0]), len(pair_similarity['mixed'][0]), low_bounds,
                         mid_bounds, high_bounds, word_window, dset, word_count)