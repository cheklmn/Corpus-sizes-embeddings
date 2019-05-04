from global_params import *
from corpus_processing import corpus_processing
import gensim.downloader as api
from gensim.models import Word2Vec

from web.evaluate import evaluate_on_all
from web.embedding import Embedding


corpus = api.load("wiki-english-20171001")

for word_count in SIZES:
    tokens = corpus_processing(corpus, word_count, RANDOMIZE_ARTICLES)
    for word_window in WORD_WINDOWS:
        model_filename = "Models/w2v_" + SIZES_MAP[word_count] + "_" + str(word_window)
        model = Word2Vec(tokens, size=300, window=word_window, workers=4)
        model.wv.save_word2vec_format(model_filename, binary=False)
        print("Model saved at " + model_filename)
        embedding_model = Embedding.from_word2vec(model_filename, binary=False)
        results = evaluate_on_all(embedding_model)
        csv_filename = "Results/w2v" + SIZES_MAP[word_count] + "_" + str(word_window) + ".csv"
        results.to_csv(csv_filename)
        print("Results saved at " + csv_filename)
