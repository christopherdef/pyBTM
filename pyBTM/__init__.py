import os
import json

# instantiate shared objects
# TODO: need way to access and edit custom resources, particularly tweet_paths

_MOD_PATH = os.path.realpath(__file__)[:os.path.realpath(__file__).rfind('/')+1]

data_paths = json.load(open(_MOD_PATH+'paths.json', 'r'))