# -*- coding: utf-8 -*-
"""
@author: Aaroh Gala
"""

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#Sample Sentence
text = "There were two sides facing three-goal deficits in the Champions League quarterfinals on Tuesday. One was widely considered to have at least a chance of going through–but it was the one that didn't that pulled off the unlikely miracle after all."

#Breaks whole text into sentences.
sentences = sent_tokenize(text)
print(sentences)

#Break whole text into words.
words =  word_tokenize(text)
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

#Stemming words - Converting into root stem of the words
ps = PorterStemmer()
words = ["Python", "Pythoner", "Pythoning", "Pythoned", "Pythonly"]
for i in words:
    print(ps.stem(i))
    
#Stemming sentences
sentences = "It is important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
words = word_tokenize(sentences)
for i in words:
    print(ps.stem(i))