'''
pyBTM usage demo
'''

import os
import sys
sys.path.append('../')

import json
import pandas as pd
import numpy as np

from pyBTM import pyBTM

import spacy
import string
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
punc = set(string.punctuation)

nlp = spacy.load('en')
nlp.disable_pipes('parser', 'tagger', 'ner')

def main():

    raw_path = 'data/tweets.csv'
    pp_path  = 'data/pp_tweets.txt'

    # quick preprocess of tweets
    print('preprocessing data')

    tweet_df = pd.read_csv(raw_path)

    # skip preprocessing if already complete
    if os.path.exists(pp_path):
        tweet_df.text = pd.Series(open(pp_path, 'r').read().split('\n'))
    else:
        tweet_df.text = tweet_df.text.apply(preprocess)
        np.savetxt(pp_path, tweet_df.text.apply(' '.join), fmt='%s')

    # initialize model
    k = 5
    alpha = 50/k
    beta = 0.01
    niter = 100
    verbose = False
    btm = pyBTM.BTM(K=k, input_path=pp_path, alpha=alpha, beta=beta, niter=niter, verbose=verbose)

    print('indexing documents')
    btm.index_documents()

    print('learning topics')
    btm.learn_topics(force=False)

    print('infering documents')
    btm.infer_documents(force=False)

    L = 5
    print('\n===',f'displaying top {L} words per topic results','===\n')
    topics = btm.get_topics(include_likelihood=False, use_words=True, L=L)
    print(json.dumps(topics, indent=4, sort_keys=True))


    print('\ngenerating coherence')
    cm_dict = btm.build_coherence_model(measures=['c_v', 'u_mass'])
    print([*map(lambda e : (e[0], e[1][1]), cm_dict.items())])

    print('\nprinting BTM info')
    print(btm.info())

def preprocess(text_line):
    '''
    Simple data preprocessing step
    '''

    # case fold
    text = text_line.casefold()

    # tokenize
    tokenized_text = nlp(text)

    # remove stopwords, puntuation, and stubs
    inclusion_condition = lambda w : w.is_alpha and\
                                     not w.is_stop and\
                                     w not in stops and\
                                     w not in punc and\
                                     len(w) >= 2

    pp_txt = [tok.lemma_ for tok in filter(lambda w : inclusion_condition(w), tokenized_text)]
    return pp_txt

if __name__ == '__main__':
    main()
