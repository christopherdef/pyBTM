import os
import sys
sys.path.append('../')

from datetime import datetime
import time
import pickle
import re

import pandas as pd
import scipy as sp
import numpy as np
import numpy.linalg as la
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt

import gensim
from gensim.models import CoherenceModel

from pyBTM import pyBTM, preprocessing, btm_utils, data_paths

def main():
    raw_path = 'data/srs_data.json'
    pp_path = 'data/srs_pp.txt'

    preprocess()

    find_best_model()

def preprocess():
    preprocessing.begin_preprocessing_tweets(read_path=raw_path, write_path=pp_path)


def find_best_model():
    # identify the models to evaluate
    K = [10,20,30,40,50]
    alpha = lambda k : 50/k
    beta = 0.01
    niter = 50
    btm_models = list([pyBTM.BTM(K=k, input=pp_path, alpha=alpha(k), beta=beta, niter=niter, verbose=True) for k in K])

    # train those models
    for btm in btm_models:
        btm.index_documents()
        btm.learn_topics()


    now_strf = lambda : datetime.now().strftime("%m-%d-%y/%H:%M:%S")

    # choose coherence measures
    coherence_measures = ['u_mass', 'c_v', 'c_uci', 'c_npmi']

    # load documents
    docs = btm_models[0].get_dwids() # documents are the same for each btm model

    # load dictionary
    print('load dictionary')
    # TODO: might be able to skip this step, or make it much shorter with a custom constructor
    #       after all, voca and dwids have most of the needed info for making the dict already
    lap_start = time.time()
    gen_dict = gensim.corpora.Dictionary.from_documents(docs)

    FILES_WRITTEN = []
    COHERENCE_VALUES = []
    i = 0

    # begin evaluating coherence for each btm model
    print('begin all eval')
    for btm in btm_models:
        print(f'{now_strf()} - evaluating {btm.param_str}')

        # load topics for the model
        gen_topics = btm.get_topics(L=-1, use_words=False, include_likelihood=False).values()

        # evaluate coherence for each model and each measure
        for measure in coherence_measures:
            print(f'{now_strf()} - \t{measure} coherence')

            # build the model
            cm = CoherenceModel(topics=gen_topics, dictionary=gen_dict, texts=docs, coherence=measure)

            # calculate coherence
            cm_value = cm.get_coherence()

            # package results
            ignore = {'dictionary', 'texts'}
            for attr in ignore:
                if hasattr(cm, attr):
                    delattr(cm, attr)

            coherence_report = {
                'btm_model'       : btm.param_str,
                'coherence_model' : cm,
                'coherence_value' : cm_value
            }

            # save model to disk
            cm_fname = btm.coherence_paths[measure]
            pickle.dump(coherence_report, open(cm_fname, 'wb'))

            FILES_WRITTEN.append(cm_fname)
            COHERENCE_VALUES.append(cm_value)

        i += 1


if __name__ == '__main__':
    main()
