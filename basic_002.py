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
    
    #Chunking and Chinking part begins
    #Generate chunking regular expression which you want to group
    chunkGram = r"""Chunk : {<RB.?>*<VB.?>*<NNP>+<NN>?} """

    #Generate chinking regular expression which you want to remove
    #In simple words, it will chunk everything and remove what we chinked/removed
    chinkGram = r"""Chunk : {<.*>+} 
                             }<VB.?|IN|DT>+{"""
    
    #Making a parser with the Regular Expression
    chunkParser = nltk.RegexpParser(chunkGram) #chunkParser = nltk.RegexpParser(chinkGram)
    #Parsing the data using parser
    chunked = chunkParser.parse(tagged)
    
    print(chunked)
    chunked.draw()
    
    #Name Entity Recognition
    nameEnt_chunk = nltk.ne_chunk(tagged, binary=True) #Setting binary = True removes all the specific name and just gives Name Entity
    print(nameEnt_chunk)
    nameEnt_chunk.draw()
'''
POS tag list:
CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
'''

'''
NE Type and Examples
ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian
'''