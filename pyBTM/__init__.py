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
