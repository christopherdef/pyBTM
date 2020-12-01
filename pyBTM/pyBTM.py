import sys
import json
import os
from subprocess import call, Popen, PIPE, check_output
import shlex
import pathlib
from pyBTM import _MOD_PATH # TODO: this sucks i hate this
from datetime import datetime
import itertools as it
from collections import Counter
import csv
import pdb

class BTM:
    '''
    Python class wrapper for running BTM
    Runs BTM code from <link>
    '''
    def __init__(self, **params):
        # shut up all the console noise if desired
        if 'verbose' in params and not params['verbose']:
            self.verbose = False
            self.stdout = open(_MOD_PATH+'btm_log.txt', 'a')
        else:
            self.verbose = True
            self.stdout = sys.stdout

        # TODO: maybe this is a bad idea. maybe make constructor very strict about the parameters that are sent in
        #       let the user package parameters in a neat json, that's not my problem
        # get last run's (or default) parameters}
        default_params = {}
        with open(_MOD_PATH+'btm_params', 'r') as f:
            default_params = json.load(f)
            if len(params) == 0:
                params = default_params

        default_params.update(params)
        self.K          = default_params['K']
        self.alpha      = default_params['alpha']
        self.beta       = default_params['beta']
        self.niter      = default_params['niter']
        self.home_dir   = _MOD_PATH # default_params['home']
        self.input_path = default_params['input']
        self.fmt        = default_params['fmt']
        self.params     = default_params

        # TODO: ugly. any recourse?
        # TODO: make visible in __repr__ and __str__
        self.param_str = f'k{self.K}_n{self.niter}_a%g_b%g' % (self.alpha, self.beta)

        # TODO: again, this is probably not something we should do on our end
        with open(_MOD_PATH+'btm_params', 'w') as f:
            json.dump(default_params, f)

        # TODO: put into constructor.
        self.save_step=501

        self.log('='*10, f'BTM INITIALIZE ({self.param_str})', '='*10)
        self.log(f"K {self.K} | alpha {self.alpha} | beta {self.beta} | niter {self.niter} | input {self.input_path}")

        # locate or create important directories
        input_fname = os.path.basename(self.input_path)
        input_fname = input_fname[:input_fname.rfind('.')]

        if 'src' in params and params['src'] is not None:
            self.src_dir = params['src']
        else:
            self.src_dir=f'{self.home_dir}BTM/src/'

        if 'output' in params and params['output'] is not None:
            self.output_dir = params['output']
        else:
            self.output_dir=f'{self.home_dir}output/{input_fname}_output/'

        self.model_dir=f'{self.output_dir}model/{self.param_str}/'
        pathlib.Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        self.coherence_types = ['u_mass', 'c_v', 'c_uci', 'c_npmi']
        self.coherence_paths = dict(zip(\
                                    self.coherence_types,
                                    [f'{self.model_dir}{self.param_str}.{ct}' for ct in self.coherence_types]\
                                  ))

        # the input docs for training
        self.doc_pt=f'{self.input_path}'

        # TODO: START - duplicated code, very bad
        # output model docs
        self.pz_pt  = f'{self.model_dir}{self.param_str}.pz'
        self.pwz_pt = f'{self.model_dir}{self.param_str}.pw_z'
        self.pzd_pt = f'{self.model_dir}{self.param_str}.pz_d'

        # vocab and indexing docs
        self.voca_pt=f'{self.output_dir}voca.txt'
        self.dwid_pt=f'{self.output_dir}doc_wids.txt'
        # TODO: END

        # set progress flags
        self.indexing_complete = os.path.exists(self.voca_pt) and os.path.exists(self.dwid_pt)
        self.learning_complete = os.path.exists(self.pz_pt) and os.path.exists(self.pwz_pt)
        self.inference_complete = os.path.exists(self.pzd_pt)

        if self.indexing_complete:
            self._load_wc()

    def _load_wc(self):
        '''
        helper for loading word and document counts
        used in init and index_documents
        '''
        self.doc_count  = self.N = int(check_output(shlex.split(f'wc -l {self.dwid_pt}')).split(b' ')[0])
        self.word_count = self.M = int(check_output(shlex.split(f'wc -l {self.voca_pt}')).split(b' ')[0])

    def index_documents(self):
        '''
        assign all words a numeric id (doc_wids) and index words in each document (vocab)
        '''
        self.log(f"========== Index Docs ({self.param_str}) ==========")

        # docs after indexing
        self.dwid_pt=f'{self.output_dir}doc_wids.txt'

        # vocabulary file
        self.voca_pt=f'{self.output_dir}voca.txt'

        # we'll never want to index all the documents twice
        if os.path.exists(self.voca_pt) and os.path.exists(self.dwid_pt):
            self.log('indexing already complete')
            self.indexing_complete = True

        # run indexing script
        else:
            index_cmd = shlex.split(f'python {self.home_dir}indexDocs.py {self.doc_pt} {self.dwid_pt} {self.voca_pt} {self.fmt}')
            self.log(' '.join(index_cmd))
            p = Popen(index_cmd, stdout=PIPE)
            self.log(p.communicate()[0].decode()) # kinda crude way to pipe subprocess output back to caller

            self.indexing_complete = True

        self._load_wc()
        self.log('indexing complete')
        return

    def learn_topics(self):
        '''
        learn parameters p(z) and p(w|z)
        '''
        if not self.indexing_complete:
            raise Exception('must index docs with BTM.index_documents() before learning parameters')

        self.log(f"========== Topic Learning ({self.param_str}) ==========")

        # might want to learn topics multiple times
        if self.learning_complete:
            r = input('learning already complete, run again?')
            if r != 'y':
                self.log('learning already complete')
                return

        # garbage code for getting the vocab size
        p = Popen(shlex.split(f'wc -l {self.voca_pt}'), stdout=PIPE)
        self.log(f'wc -l {self.voca_pt}')
        W = int(p.communicate()[0].decode().split(' ')[0]) # vocabulary size

        # make the BTM source code
        p = Popen(shlex.split(f'make -C {self.src_dir}'), stdout=PIPE)
        self.log(p.communicate()[0].decode())

        # run btm est
        btm_est_cmd = shlex.split(f'{self.src_dir}btm est '+\
                                  f'{self.K} {W} {self.alpha} {self.beta} {self.niter} {self.save_step} '+\
                                  f'{self.dwid_pt} {self.model_dir}')
        self.log(' '.join(btm_est_cmd))
        p = Popen(btm_est_cmd, stdout=PIPE)
        self.log(p.communicate()[0].decode())

        self.learning_complete = True
        self.log('learning complete')
        return


    def infer_documents(self):
        '''
        infer p(z|d) for each doc
        '''
        if not self.indexing_complete or not self.learning_complete:
            raise Exception('must index documents, then learn models before inferring topic distribution per document\n'+\
                            'run .index_documents() and .learn_topics() before inference')

        # might want to run inference multiple times
        if self.inference_complete:
            r = input('inference already complete, run again?')
            if r != 'y':
                self.log('inference already complete')
                return

        self.log(f"========== Infer P(z|d) ({self.param_str}) ==========")

        btm_inf_cmd = shlex.split(f'{self.src_dir}btm inf sum_b '+\
                                  f'{self.K} {self.alpha} {self.beta} {self.niter} '+\
                                  f'{self.dwid_pt} {self.model_dir}')
        self.log(' '.join(btm_inf_cmd))

        p = Popen(btm_inf_cmd, stdout=PIPE)
        self.log(p.communicate()[0].decode())

        self.inference_complete = True
        self.log('inference complete')
        return

    def display_topics(self):
        '''
        output top words of each topic
        '''
        self.log(f"================ Topic Display ({self.param_str}) =============")

        topic_display_cmd = shlex.split(f'python {self.home_dir}topicDisplay.py '+\
                                        f'{self.model_dir} {self.K} {self.voca_pt} '+\
                                        f'{self.pz_pt} {self.pwz_pt}')
        self.log(" ".join(topic_display_cmd))

        p = Popen(topic_display_cmd, stdout=PIPE)
        self.log(p.communicate()[0].decode())

    # TODO implement lazy
    def get_topics(self, L=10, use_words=True, include_likelihood=True):
        '''
        Connects the topic distribution (pz) with the word-by-topic distribution (pw_z) and the vocab dict
        and returns a 3D array where each row is a topic and each entry is a [word, word_likelihood] pair
        If include_likelihood is false, returns a 2D array where each row is a topic and each entry is a word

        @param L - number of words per topic to return in array; L<=0 --> return all words in topic
        @param use_words - if False, shows only the word_id in the array
        @param include_likelihood - if False, each entry returned is a word rather than a [word, word_likelihood] pair
        @param lazy - if True, each entry returned is a generator rather than full array
        @return 3D array of top wids/words per topic along paired with those words' likelihoods
        '''
        vocab = self.get_vocab()
        pz = self.get_pz()
        pw_z = self.get_pwz(lazy=True)

        i = 0
        top_words_per_topic = {}
        for topic_line in pw_z:
            z = pz[i]

            # sort the w|z values
            sorted_wz = sorted(enumerate(topic_line), key=lambda wid_wz : float(wid_wz[1]), reverse=True)
            top_wz = sorted_wz

            # slice if needed
            if L > 0:
                top_wz = sorted_wz[:L]

            # TODO: messy, hard to read, pls fix
            # apply parameters use_words and include_likelihood to the top words in topic z
            apply_use_words = lambda wid : str(wid) if not use_words else vocab[str(wid)]
            apply_include_likelihood = lambda wid_wz_pair : wid_wz_pair if include_likelihood else wid_wz_pair[0]
            top_wz = list([\
                           apply_include_likelihood([\
                                                     apply_use_words(wid),\
                                                     str(wz)\
                                                    ])
                           for wid, wz in top_wz])

              # old imp of above
#             if not use_words:
#                 top_wz = list([[str(wid), str(wz)] for wid, wz in top_wz])
#             # map wids to words if needed
#             else:
#                 top_wz = list([[vocab[str(wid)], str(wz)] for wid, wz in top_wz])

            top_words_per_topic[z] = top_wz
            i+=1

        # returns array of top words per topic along with those words' likelihoods
        return top_words_per_topic

    def get_pz(self):
        if not self.learning_complete:
            raise Exception('topic learning has not been completed')

        pz = open(self.pz_pt, 'r').read().rstrip().split(' ')

        return pz

    def get_pwz(self, lazy=True, dtype=str):
        if not self.learning_complete:
            raise Exception('topic learning has not been completed')
        pwz_file = open(self.pwz_pt, 'r')
        process_pwz_line = lambda line : list(dtype(e) for e in line.rstrip().split(' '))

        if lazy:
            return self._lazy_load(pwz_file, process_pwz_line)
        else:
            pwz = []
            for topic_line in pwz_file:
                word_likelihood = process_pwz_line(topic_line)
                pwz.append(word_likelihood)

            return pwz

    def get_doc_info(self, lazy=True, query=None):
        '''
        Loads the .info file associated with the input data
        This file should contain, in csv format, all the metadata associated with each document

        '''

        fp = self.input_path+'.info'
        f  = open(fp, 'r')
        reader = csv.reader(f, delimiter=',')

        removed_docs = self.get_removed_docs()

        def _info_lazy_load(reader, removed_docs, query):
            sentinel_lazy = object()
            while True:
                r = next(reader, sentinel_lazy)
                if r is sentinel_lazy:
                    f.close()
                    return
                if reader.line_num in removed_docs:
                    continue
                if query is None:
                    yield r
                else:
                    yield query(r)



        sentinel = object()
        data_info = []
        line_loader = _info_lazy_load(reader, removed_docs, query)

        if lazy:
            return line_loader
        else:
            while True:
                loaded_line = next(line_loader, sentinel)
                if loaded_line is sentinel:
                    break

                data_info.append(loaded_line)

            f.close()
            return data_info

    # TODO: naive soln, integrate some memmap or dask stuff later
    def _lazy_load(self, f, process_line):
        for line in f:
            yield process_line(line)

    # TODO: implement lazy
    def get_pzd(self, lazy=True):
        if not self.inference_complete:
            raise Exception('inference has not been completed')

        pzd_file = open(self.pzd_pt, 'r')

        if lazy:
            process_pzd_line = lambda line : line.rstrip().split(' ')
            return self._lazy_load(pzd_file, process_pzd_line)

        pzd = []
        for doc_line in pzd_file:
            topic_likelihood = doc_line.rstrip().split(' ')
            pzd.append(topic_likelihood)

        return pzd

    def get_removed_docs(self):
        removed_docs_pt = f'{self.output_dir}removed_docs.txt'

        if not os.path.exists(removed_docs_pt):
            return {}

        removed_docs_file = open(removed_docs_pt, 'r')

        removed_docs = set()
        for line in removed_docs_file:
            removed_docs.add(int(line.rstrip()))

        removed_docs_file.close()

        return removed_docs


    def get_word_count(self):
        if not self.indexing_complete:
            raise Exception('indexing has not been completed')

        count_words_cmd = shlex.split(f'wc -l {self.voca_pt}')
        p = Popen(count_words_cmd, stdout=PIPE)
        M = int(p.communicate()[0].decode().split(' ')[0])
        return M

    def get_doc_count(self):
        count_words_cmd = shlex.split(f'wc -l {self.dwid_pt}')
        p = Popen(count_words_cmd, stdout=PIPE)
        N = int(p.communicate()[0].decode().split(' ')[0])
        return N

    def _get_output_iter(self, path):
        '''
        returns a line-by-line lazy iterator for whatever data is at the path
        '''
        with open(path,'r') as f:
            for line in f:
                yield line

    def log(self, *msg, **kwds):
        print(datetime.now().strftime('%m-%d-%y/%H:%M:%S\t'), *msg, file=self.stdout, **kwds)

    # TODO implement lazy
    def get_dwids(self, lazy=False):
        if not self.indexing_complete:
            raise Exception('indexing has not been completed')

        self.log('unpacking doc_wids')
        return list(l.strip().split(' ') for l in open(self.dwid_pt, 'r'))

    def get_word_freq(self, bow_docs=None):
        if not self.indexing_complete:
            raise Exception('indexing has not been completed')

        if bow_docs is None:
            bow_docs = self.get_dwids()
        self.log('generating word frequency distribution')
        return Counter(it.chain(*bow_docs))

    def get_vocab(self):
        if not self.indexing_complete:
            raise Exception('indexing has not been completed')

        self.log('unpacking vocab')
        return dict((tuple([wid_w[0], wid_w[1]]) for wid_w in\
                          [line.strip().split('\t') for line in open(self.voca_pt, 'r')]))



    # TODO: implement coherence stuff
    def build_coherence_model(self, coherence_type='c_v'):
        if os.path.exists(self.coherence_paths[coherence_type]):
            r = input('inference already complete, run again?')
            if r != 'y':
                self.log('inference already complete')
                return
        pass
    def save_coherence_model(self, cm):
        pass
    def load_coherence_model(self):
        pass


    def remove_extremes(self, T_lo=10, T_hi=50, T_lo_freq=1, bow_docs=None):
        '''
        Remove extremities from the vocabulary

        @param T_lo -
        @param T_hi -
        @param T_lo_freq -
        @param bow_docs -
        '''
        if not self.indexing_complete:
            raise Exception('must index docs with BTM.index_documents() before removing infrequent words')

        if os.path.exists(self.voca_pt+'.old') or os.path.exists(self.dwid_pt+'.old'):
            self.log(f'Clean up the output directory from previous run first! \n\'rm {self.output_dir}*.old\'\nAborting')
            return

        self.log('removing infrequent words from dwids')

        # load doc_wids
        if bow_docs is None:
            bow_docs = self.get_dwids()

        # get word frequencies
        word_freq = self.get_word_freq(bow_docs)

        word_freq = self.get_word_freq()
        ordered_word_freq = word_freq.most_common()
        most_freq = list(wf[0] for wf in ordered_word_freq[:T_hi]) if T_hi > 0 else []

        # TODO: least_freq doesn't account for words on the bottom already being cut by T_low_freq, maybe fix that sometime
        least_freq = list(wf[0] for wf in ordered_word_freq[-T_lo:]) if T_lo > 0 else []

        blacklist = set.union(set(most_freq+least_freq),\
                              set(it.filterfalse(lambda w : word_freq[w] > T_lo_freq, word_freq.keys())))

        # clean up for memory reservation
        # TODO: is this necessary?
        del ordered_word_freq
        del most_freq
        del least_freq
        del word_freq

        old_vocab = self.get_vocab()
        new_vocab = {}
        old_id_to_new_id = {}
        j = 0
        for wid in old_vocab:
            if wid in blacklist:
                pass
            else:
                assert j not in new_vocab,\
                       f'{j} already logged as "{new_vocab[j]}", trying to duplicate as "{old_vocab[wid]}"'
                new_vocab[str(j)] = old_vocab[wid]
                old_id_to_new_id[wid] = str(j)
                j += 1

        # cleanup
        del blacklist

        # rewrite vocab file
        os.rename(self.voca_pt, self.voca_pt+'.old')
        new_voca_file = open(self.voca_pt, 'w')
        for wid, w in new_vocab.items():
            print(f'{wid}\t{w}', file=new_voca_file)

        # send old dwids path away
        os.rename(self.dwid_pt, self.dwid_pt+'.old')

        # create new file for new dwids
        open(self.dwid_pt, 'a').close()
        new_dwids_file = open(self.dwid_pt, 'w')

        # create new file for docs that will be removed during this process
        self.removed_docs_pt = f'{self.output_dir}removed_docs.txt'
        removed_docs_file = open(self.removed_docs_pt, 'w')

        self.log('writing processed lines')
        line_fmt = lambda l : ' '.join(l)

        N = 1e5
        i = 0
        j = 0

        # filter low-freq words, format bow --> text, then write to new file
        for doc in bow_docs:
            new_doc = list()
            for wid in doc:
                if wid in old_id_to_new_id:
                    new_wid = old_id_to_new_id[wid]
                    new_doc.append(new_wid)
            #new_doc = list(it.filterfalse(lambda wid : wid not in new_vocab, doc))
            if len(new_doc) == 0:
                index_of_doc_to_be_deleted = i
                print(index_of_doc_to_be_deleted, file=removed_docs_file)
            else:
                j += 1
                print(' '.join(new_doc), file=new_dwids_file)

            i += 1
            if i != 0 and i%N == 0:
                if self.verbose:
                    self.log(f'written {i} docs')
                    pass
                    #print('#', end='')

        #print(new_vocab)

        words_removed_count = len(old_vocab)-len(new_vocab)
        del old_vocab

        self.log(f'Reduced size from {i} docs to {j} docs! Removed {words_removed_count} words!')
        removed_docs_file.close()
        new_dwids_file.close()
        new_voca_file.close()
