import gensim.downloader as api
import json
import random


corpus = api.load("patent-2017")

item = iter(corpus)
current = next(item)
while random.randint(0, 100) % 70 != 0:
    current = next(item)
js = json.dumps(current, indent=4)
print(js)
