from corpus_sizes.global_params import *
from corpus_sizes.corpus_processing import corpus_processing

import scipy
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models import KeyedVectors

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

    word_freqs = [(word, c[word]) for word in vocabulary if c[word] > 0]
    out_of_vocabulary = [(word, c[word]) for word in vocabulary if c[word] == 0]
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

    return high, mid, low, high_bounds, mid_bounds, low_bounds, out_of_vocabulary


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


def save_results(embedding_type, gen_similarity, low_similarity, mid_similarity, high_similarity, mixed_similarity,
                 out_of_vocab, low_count, mid_count, high_count, mixed_count, low_bounds, mid_bounds, high_bounds,
                 dim, ww, dataset_name, corpus_size):
    method = "SG" if SKIP_GRAMS else "CBOW"
    cross_sent = "Yes" if CROSS_SENTENCE else "No"
    results_file = './Results/Results.pickle'

    if os.path.isfile(results_file):
        df = pd.read_pickle(results_file)
    else:
        df = pd.DataFrame()
    df = df.append(pd.DataFrame({
        "Embedding": embedding_type,
        "Method": method,
        "Time": datetime.datetime.now(),
        "Dimension": dim,
        "Window": ww,
        "Word count": corpus_size,
        "Sampling": SAMPLING,
        "Cross-sentence": cross_sent,
        "Epochs": str(EPOCHS),
        "Dataset": dataset_name,
        "Out of vocabulary": out_of_vocab,
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


def evaluate_w2v(data, current_model, similarity_pairs):

    general_similarity = evaluate_similarity(current_model, data.X, data.y)
    low_similarity = evaluate_similarity(current_model,
                                               np.asarray(similarity_pairs['low'][0]),
                                               np.asarray(similarity_pairs['low'][1])
                                               )
    mid_similarity = evaluate_similarity(current_model,
                                               np.asarray(similarity_pairs['mid'][0]),
                                               np.asarray(similarity_pairs['mid'][1])
                                               )
    high_similarity = evaluate_similarity(current_model,
                                                np.asarray(similarity_pairs['high'][0]),
                                                np.asarray(similarity_pairs['high'][1])
                                                )
    mixed_similarity = evaluate_similarity(current_model,
                                                 np.asarray(similarity_pairs['mixed'][0]),
                                                 np.asarray(similarity_pairs['mixed'][1])
                                                 )
    return general_similarity, low_similarity, mid_similarity, high_similarity, mixed_similarity


def evaluate_fasttext(current_model, X, y):
    oov_pairs = 0
    for query in X:
        out_of_vocabulary = False
        for query_word in query:
            if query_word not in current_model.wv.vocab:
                out_of_vocabulary = True
        if out_of_vocabulary:
            oov_pairs += 1;

    A = np.vstack(current_model.wv[word] for word in X[:, 0])
    B = np.vstack(current_model.wv[word] for word in X[:, 1])
    scores = np.array([v1.dot(v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    return scipy.stats.spearmanr(scores, y).correlation, oov_pairs


def evaluation(datset, data, sentences, vocabulary, emb_model, model_type, model_dimension, model_window, model_wordcount):
    high, mid, low, high_bounds, mid_bounds, low_bounds, out_of_voc = single_word_bins(sentences, vocabulary)
    pair_similarity = pair_bins(low, mid, high, data)

    if model_type =='w2v':
        model_general_similarity, model_low_similarity, model_mid_similarity, \
            model_high_similarity, model_mixed_similarity = evaluate_w2v(data, emb_model, pair_similarity)
        save_results('w2v', model_general_similarity, model_low_similarity, model_mid_similarity, model_high_similarity,
                     model_mixed_similarity, len(out_of_voc), len(pair_similarity['low'][0]),
                     len(pair_similarity['mid'][0]), len(pair_similarity['high'][0]),
                     len(pair_similarity['mixed'][0]), low_bounds, mid_bounds, high_bounds, model_dimension,
                     model_window, datset, model_wordcount)
    else:
        model_general_similarity, out_of_voc = evaluate_fasttext(emb_model, data.X, data.y)
        model_low_similarity, buffer = evaluate_fasttext(emb_model,
                                             np.asarray(pair_similarity['low'][0]),
                                             np.asarray(pair_similarity['low'][1])
                                             )
        model_mid_similarity, buffer = evaluate_fasttext(emb_model,
                                             np.asarray(pair_similarity['mid'][0]),
                                             np.asarray(pair_similarity['mid'][1])
                                             )
        model_high_similarity, buffer = evaluate_fasttext(emb_model,
                                              np.asarray(pair_similarity['high'][0]),
                                              np.asarray(pair_similarity['high'][1])
                                              )
        model_mixed_similarity, buffer = evaluate_fasttext(emb_model,
                                               np.asarray(pair_similarity['mixed'][0]),
                                               np.asarray(pair_similarity['mixed'][1])
                                               )
        save_results('fasttext', model_general_similarity, model_low_similarity, model_mid_similarity, model_high_similarity,
                     model_mixed_similarity, out_of_voc, len(pair_similarity['low'][0]),
                     len(pair_similarity['mid'][0]), len(pair_similarity['high'][0]),
                     len(pair_similarity['mixed'][0]), low_bounds, mid_bounds, high_bounds, model_dimension,
                     model_window, datset, model_wordcount)


corpus = api.load("wiki-english-20171001")


for word_count in SIZES:
    sents = corpus_processing(corpus, word_count, RANDOMIZE_ARTICLES)
    for embedding in USE_EMBEDDING:
        for dimension in DIMENSION:
            for word_window in WORD_WINDOWS:
                model_filename = "./Models/" + embedding + "_" + SG_MAP[SKIP_GRAMS] + "_" \
                                 + SIZES_MAP[word_count] + "_" + str(dimension) + "_" + str(word_window) + "_" \
                                 + SAMPLING + "_" + str(EPOCHS)
                vocab_filename = "./Vocab/" + embedding + "_" + SG_MAP[SKIP_GRAMS] + "_" \
                                 + SIZES_MAP[word_count] + "_" + str(dimension) + "_" + str(word_window) + "_" \
                                 + SAMPLING + "_" + str(EPOCHS)

                if not USE_CACHED:
                    skip_grams = 1 if SKIP_GRAMS else 0
                    softmax = 1 if SAMPLING == 'hs' else 0
                    negative_sample = 10 if SAMPLING == 'ns' else 0
                    if embedding == 'w2v':
                        model = Word2Vec(sents, size = dimension, window = word_window, workers=3,
                                         sg = skip_grams, hs = softmax, negative = negative_sample, iter = EPOCHS)
                        model.wv.save_word2vec_format(model_filename, binary=False)
                        model.vocabulary.save(vocab_filename)
                    else:
                        model = FastText(sg = skip_grams, hs = softmax, size = dimension, window = word_window,
                                         workers = 3, negative = negative_sample)
                        model.build_vocab(sentences = sents)
                        model.train(sentences = sents, total_examples = len(sents), epochs = EPOCHS)
                        model.save(model_filename)
                        FastText.load
                        # model.vocabulary.save(vocab_filename)

                    print("Model saved at " + model_filename)

                embedding_model = []
                if embedding == 'w2v':
                    embedding_model = Embedding.from_word2vec(model_filename)
                else:
                    embedding_model = FastText.load(model_filename)

                for dset in USE_DATASETS:
                    dataset = get_dataset(dset)
                    vocab = set()
                    for pair in dataset.X:
                        vocab.add(pair[0])
                        vocab.add(pair[1])

                    evaluation(dset, dataset, sents, vocab, embedding_model, embedding, dimension, word_window, word_count)
                    # high, mid, low, high_bounds, mid_bounds, low_bounds, out_of_voc = single_word_bins(sents, vocab)
                    # pair_similarity = pair_bins(low, mid, high, dataset)
                    #
                    # model_general_similarity = evaluate_similarity(model, dataset.X, dataset.y)
                    # model_low_similarity = evaluate_similarity(model,
                    #                                            np.asarray(pair_similarity['low'][0]),
                    #                                            np.asarray(pair_similarity['low'][1])
                    #                                            )
                    # model_mid_similarity = evaluate_similarity(model,
                    #                                            np.asarray(pair_similarity['mid'][0]),
                    #                                            np.asarray(pair_similarity['mid'][1])
                    #                                            )
                    # model_high_similarity = evaluate_similarity(model,
                    #                                             np.asarray(pair_similarity['high'][0]),
                    #                                             np.asarray(pair_similarity['high'][1])
                    #                                             )
                    # model_mixed_similarity = evaluate_similarity(model,
                    #                                              np.asarray(pair_similarity['mixed'][0]),
                    #                                              np.asarray(pair_similarity['mixed'][1])
                    #                                              )
                    # save_results(embedding, model_general_similarity, model_low_similarity, model_mid_similarity,
                    #              model_high_similarity,
                    #              model_mixed_similarity, len(out_of_voc), len(pair_similarity['low'][0]),
                    #              len(pair_similarity['mid'][0]), len(pair_similarity['high'][0]),
                    #              len(pair_similarity['mixed'][0]), low_bounds, mid_bounds, high_bounds, dimension, word_window,
                    #              dset, word_count)
                    # if embedding == 'w2v':
                    #     evaluate_w2v(dset, dataset, sents, vocab, embedding_model, dimension, word_window, word_count)
                    # else
                    #     evaluate_fasttext(dset, dataset, sents, vocab, embedding_model, dimension, word_window, word_count)