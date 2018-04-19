# -*- coding: utf-8 -*-
"""
@author: Aaroh
"""

from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize, sent_tokenize

#Fetch data from corpus database
sample_text = gutenberg.raw("bible-kjv.txt")

#Sentence Tokenizer
sentences = sent_tokenize(sample_text)
print(sentences)

#Word tokenizer
words = word_tokenize(sample_text)
print(words)