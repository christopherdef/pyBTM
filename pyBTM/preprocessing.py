import json
import pickle
import time
import os

from pyBTM import btm_utils, data_paths, _MOD_PATH
from pyBTM.btm_utils import get_json_enumerator, get_file_length

import string
from dateutil.parser import parse
import regex as re

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import nltk
import spacy

# TODO: horrifyingly unstable
nlp = spacy.load("en")
nlp.disable_pipes('parser', 'tagger', 'ner')

# TODO: extremely inflexible, hard to use, and dangerous
def begin_preprocessing_tweets(read_path, write_path, debug_thresh=-1, fmt='json'):
    '''
    Begins preprocessing all tweets located at read_path
    Writes processed tweets to file at write_path
    @return - (j, delta)
        j - the number of tweets processed at read_path
        delta - time elapsed during processing in seconds
    '''
    READ_PATH = read_path
    WRITE_PATH = write_path
    INFO_WRITE_PATH = write_path+'.info'

    tw_enum = btm_utils.get_json_enumerator(READ_PATH)
    N = btm_utils.get_file_length(READ_PATH)
    i = 0
    current_prog = 0

    # TODO: should be calculated based on N. Too low and it won't move, too high and stdout io will bog everything down
    PERC_SCALE = 10_000
    perc = lambda i,n : (i/n)*PERC_SCALE

    if not os.path.exists(WRITE_PATH):
        open(WRITE_PATH, 'w').close()

    # fail-safe
    # TODO: not good enough imo
    WP_SIZE = btm_utils.get_file_length(WRITE_PATH)
    r = input(f'About to erase *ALL* data in {WRITE_PATH} File contains {WP_SIZE} bytes of data.\nContinue? (y/n)')
    if r not in ['y', 'Y']:
        print('Operation Cancelled')
        return

    # erase everything in the file at WRITE_PATH
    f_out = open(WRITE_PATH, 'w')
    f_out_info = open(INFO_WRITE_PATH, 'w')


    # Preprocessing
    start = time.time()
    stopwords = set(nltk.corpus.stopwords.words('english')+nltk.corpus.stopwords.words('russian'))
    punctuation = set(string.punctuation)
    stopwords, punctuation, regx = load_custom_resources(stopwords, punctuation)
    N = btm_utils.get_file_length(READ_PATH)

    Nb = int(10e5)
    j = 0
    debug_thresh = -1
    print('='*10,'BEGIN PREPROCESSING','='*10)

    while i < N:
        i+=1

        # debugging break
        j += 1
        if debug_thresh > 0 and j > debug_thresh:
            break

        tw,i = next(tw_enum)

        # TODO: should NOT be hardcoded, maybe circle back to json query idea?
        # ignore all non-english tweets
        if 'lang' not in tw or tw['lang'] != 'en':
            j -= 1
            continue

        # get preprocessed text
        tw_pp = preprocess_tweet(tw, nlp, stopwords, punctuation, regx)

        # ignore tweets with no lemmas
        if len(tw_pp) > 0:
            date_str = parse(tw['created_at']).strftime('%m-%d-%y')
            # pair each tweet with its id and the date it was made
            info = (tw['id_str'], date_str)
            print(' '.join(tw_pp), file=f_out)
            print(','.join(info), file=f_out_info)


        if i % 5_000 == 0:
            current_prog = int(perc(i,N))
            print('%.2f%%'%(current_prog/(PERC_SCALE/100)), end='\r')
            #print('#'*int(current_prog/10), end=' '*(10-int(current_prog/10))+'|\r')

    # write last bit of buffer
    print('100%')

    # cleanup
    f_out.close()

    print('='*10,'COMPLETE','='*10)
    print(f'Preprocessed {j}')
    delta = time.time()-start
    return (j, delta)

def preprocess_tweet(tw, nlp, stopwords, punctuation, regx, min_word_len=2):
    '''
    Preprocesses a single tweet with the following steps:
        1. case folding
        2. tokenization
        3. lemmatization
        4. stub filtering
        5. stopword filtering
        6. punctuation filtering
        7. regular expression filtering
    @param tw - tweet in json format
    @param nlp - spacy tokenizer
    @param stopwords - set of stopwords to exclude
    @param punctuation - set of punctuation to exclude
    @param regx - list of regular expressions, matches are excluded
    @param min_word_len - lemmatized words shorter than this are excluded
    @return a list of lemmatized words in the tweet
    '''
    text = tw['text'] #TODO: not robust to full_text or its variants

    # case fold
    text = text.lower()

    # tokenize
    tokenized_text = nlp(text)

    # lemmatize
    lem_text = []
    for word in tokenized_text:
        lem = word.lemma_.strip()

        # process the lemma
        if lem:

            # remove stop words, punctuation, and words shorter than min_word_len
            if len(lem) <= min_word_len or lem in stopwords or lem in punctuation:
                continue

            # apply regex, break early if a match is found
            i = 0
            while i < len(regx):
                if regx[i].search(lem) != None:
                    break
                i += 1

            # do not add lemma if loop was broken early
            if i != len(regx):
                continue

            # add to lemmas if all tests were passed
            lem_text.append(lem)
    return lem_text

def load_custom_resources(stop=set(), punc=set(), regx=[], dir_path=_MOD_PATH+'resources/'):
    '''
    Returns sets of known patterns to be used in preprocessing
    @param stop - set of stopwords to exclude
    @param punc - set of punctuation to exclude
    @param regx - list of regular expressions, matches are excluded
    '''

    # load custom stopwords, punctuation
    stop = stop.union(stop, set(open(dir_path+'stopwords').read().split('\n')))
    punc = punc.union(punc, set(open(dir_path+'punctuation').read().split('\n')))

    # load and compile regex
    for line in open(dir_path+'regex').read().split('\n'):
        if line == '':
            continue
        if r'//#' in line:
            line = line[:line.rfind(r'//#')]
        regx.append(re.compile(line.strip(), re.UNICODE))

    return stop, punc, regx

def _write_buffer(buf, path):
    '''
    Simple helper for writing line-separated buffers to files
    '''
    with open(path, 'a') as file:
        file.write('\n'.join(buf))
