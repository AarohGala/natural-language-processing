# -*- coding: utf-8 -*-
"""
@author: Aaroh
"""

"""
The major difference between these is, as you saw earlier, stemming can often create non-existent words, whereas lemmas are actual words.
So, your root stem, meaning the word you end up with, is not something you can just look up in a dictionary, but you can look up a lemma.
"""

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

#Sample Sentence
sample_text = "This module brings together a variety of NLTK functionality for text analysis, and provides simple, interactive interfaces. Functionality includes: concordancing, collocation discovery, regular expression search over tokenized strings, and distributional similarity."

#Break whole text into words.
words =  word_tokenize(sample_text)
print(words)

#Get all the stop words of language English.
stop_words = set(stopwords.words("english"))
print(stop_words)

#Remove all the stop words from the text
filtered_list = []
for i in words:
    if i not in stop_words:
        filtered_list.append(i)
print(filtered_list)

#Stemming of words
print("Stemming")
ps = PorterStemmer()
for i in filtered_list:
    print(ps.stem(i))
    
#Lemmatization of words
print("Lemmatization")
lemmatizer = WordNetLemmatizer()
for i in filtered_list:
    #If pos tag is not given then it will try to find nearest Dict Noun
    print(lemmatizer.lemmatize(i))
    #It tried to find nearest Part of Speech given
    print(lemmatizer.lemmatize(i, pos="a"))
