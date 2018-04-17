# -*- coding: utf-8 -*-
"""
@author: Aaroh
"""

import nltk
#Courpus has lot of text data which we can used
from nltk.corpus import state_union
#Customised Sentence Tokenizer
from nltk.tokenize import PunktSentenceTokenizer

#Extract data from corpus
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

#Train the Punkt Tokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
#Sentence Tokenizer Model
sentences = custom_sent_tokenizer.tokenize(sample_text)

for i in sentences:
    #Word tokenizer
    words = nltk.word_tokenize(i)
    #Part of Speech tagging
    tagged = nltk.pos_tag(words)
    print(tagged)