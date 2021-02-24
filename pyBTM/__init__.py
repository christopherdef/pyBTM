import sys
import os
import json
from datetime import datetime

# instantiate shared objects
# TODO: need way to access and edit custom resources, particularly tweet_paths


class Logger:
    def __init__(self, verbose=False, log_pt=None):
        self.verbose = verbose
        log_pt = log_pt or os.path.join(os.path.dirname(__file__), 'btm_log.txt')

        if not os.path.exists(log_pt):
            open(log_pt, 'w+').close()

        self.log_file = open(log_pt, 'a')

    def log(self, *msg, **kwds):
        message = datetime.now().strftime('%m-%d-%y/%H:%M:%S\t') + ' '.join([*msg])

        if 'verbose' in kwds and kwds['verbose']:
            print(message, file=sys.stdout, **kwds)

        print(message, file=self.log_file, **kwds)


class Indexer:
    def __init__(self):
        pass

    
    @staticmethod
    def indexFile(pt, res_pt):
        w2id = {}
        print('index file: '+str(pt))
        i = 0
        wf = open(res_pt, 'w', encoding='utf8')
        for l in open(pt, encoding='utf8'):
            i+=1
            if i % 10_000 == 0:
                print(i, end='\r')

            ws = l.strip().split()
            for w in ws:
                if w not in w2id:
                    w2id[w] = len(w2id)

            wids = [w2id[w] for w in ws]
            print(' '.join(map(str, wids)), file=wf)

        return w2id

    @staticmethod
    def write_w2id(w2id, res_pt):
        print('vocab write to: '+str(res_pt))
        wf = open(res_pt, 'w', encoding='utf8')
        for w, wid in sorted(w2id.items(), key=lambda d:d[1]):
            print('%d\t%s' % (wid, w), file=wf)
