# configure spacy and nltk
from setuptools import setup

setup(name='pyBTM',
      version='0.1',
      description='Python wrapper for the Biterm Topic Model implemented in C',
      url='',
      author='Christopher de Freitas',
      author_email='christopherdef@gmail.com',
      license='MIT',
      packages=['pyBTM'],
      zip_safe=False)

import nltk
nltk.download('stopwords')

import spacy
spacy.cli.download('en_core_web_sm')







