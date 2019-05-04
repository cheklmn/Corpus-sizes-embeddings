# Word counts for models
SIZES = [
    1000000,
    5000000,
    10000000,
    50000000,
    100000000,
    500000000,
    1000000000
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
    3,
    5,
    10
]

# Use models made before
USE_CACHED = True

# Randomize the articles' order and their sentences
RANDOMIZE_ARTICLES = True