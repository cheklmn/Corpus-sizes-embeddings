# Corpus-sizes-embeddings
The purpose of this project is to research the influence of corpora sizes and 
term frequencies on the accuracy of word embeddings trained on those corpora

## Introduction
In the application, the text from english Wikipedia corpus was used. 
The idea is to train models on corpus slices of different size and then evaluate the model on various datasets
To test word frequencies influence, the following approach is taken:
* Word pairs from a dataset are 
Low, Mid, High and Mixed, based on the frequency of words in the pair

To determine the effect corpus size has on the word embedding, 

## Requirements
The project requires Python 2.7 and following packages to be installed:
* nltk
* gensim
* pandas
* tqdm
* matplotlib
* word-embedding benchmarks from https://github.com/kudkudak/word-embeddings-benchmarks

## Usage
Install packages from environment.ymd.
Configure project by editing global_params.py

## Results
Currently result include only Spearman coefficient on various datasets, including both similarity and analogy (see [Results/w2v5M_10.csv](Results/w2v5M_10.csv) for example

## Next steps
* Move to Python 3
* Include analogy datasets for testing word frequencies
* Add visualization of results
* Include different embedding settings
