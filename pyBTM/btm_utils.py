import os
import json

def get_json_enumerator(path):
    '''
    Lazy iterator for newline separated json entities located at path
    @return - (tweet, fpos)
        tweet - dict of the tweet data
        fpos - number of bytes enumerated through the file so far
    '''
    with open(path, 'rb') as file:
        fpos = 0

        for line in file:
            if len(line) <= 1:
                break
            tweet = json.loads(line)
            fpos += len(line)
            yield (tweet,fpos)

        return None

def get_file_length(path):
    '''
    Returns the length in bytes of a local file
    '''
    with open(path, 'r') as file:
        return os.fstat(file.fileno()).st_size
