# Embedding model
USE_EMBEDDING = [
    #'ftext',
    'w2v'
]

# Word counts for models
SIZES = [
    1000000,
    5000000,
    10000000,
    #50000000,
    #100000000,
    #500000000,
    #1000000000
]
SIZES_MAP = {
       1000000: "1M"  ,
       5000000: "5M"  ,
      10000000: "10M" ,
      50000000: "50M" ,
     100000000: "100M",
     500000000: "500M",
    1000000000: "1B"  ,
}

# Word window size
WORD_WINDOWS = [
    2,
    5,
    10,
    #15
]

#Vector dimension size
DIMENSION = [
    #100,
    300
]

# Use models made before
USE_CACHED = False

# Randomize the articles' order and their sentences
RANDOMIZE_ARTICLES = True

# Datasets that are going to be used for evaluation
USE_DATASETS = [
    #'MEN',
    #'MTurk',
    #'RG65',
    #'RW',
    #'SimLex-999',
    #'TR9856',
    #'WS353',
    'Google',
    'MSR'
]

# Use skip-grams(True) or CBOW(False):
SKIP_GRAMS = False

SG_MAP = {
    True: "SG",
    False: "CB"
}

# Use negative sampling or hierarchical softmax
SAMPLING = 'ns' # 'hs' or 'ns'

# Number of iterations for training
EPOCHS = 10

# Use cross-sentence word windows
CROSS_SENTENCE = True