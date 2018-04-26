# -*- coding: utf-8 -*-
"""
@author: Aaroh
"""

import nltk
import random
#Getting moview reviews data from corpus
from nltk.corpus import movie_reviews

#Creating a list of tuples with reviews and its category as good or bad
documents = []
for category in movie_reviews.categories():
    for fieldid in movie_reviews.fileids(category):
        documents.append((movie_reviews.words(fieldid), category))
print(documents)

#Shuffling the tuples in the list
random.shuffle(documents)

#Getting all the words from the movie review documents
all_words = []
for word in movie_reviews.words():
    #converting to lowercase
    all_words.append(word.lower())

#Computing the frequency of all the words in all_word list
all_words = nltk.FreqDist(all_words)
#Printing 15 most commonly used words 
print(all_words.most_common(45))

#printing number of occurence of word "good"
print(all_words["good"])

word_features = list(all_words.keys())[:3000]
print(word_features)

def find_features(document):
    print(document)
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    
    return features



featureSet = [(find_features(rev), category) for (rev,category) in documents]
print(featureSet)

trainSet = featureSet[:1500]
testSet = featureSet[1500:]

classifys = nltk.NaiveBayesClassifier.train(trainSet)

print("Naive Bayes Accuracy:", (nltk.classify.accuracy(classifys, testSet))*100)
classifys.show_most_informative_features(20)


'''
OUTPUT:
('Naive Bayes Accuracy:', 71.39999999999999)
Most Informative Features
               insulting = True              neg : pos    =     12.3 : 1.0
                  doubts = True              pos : neg    =      9.0 : 1.0
              moderately = True              neg : pos    =      7.6 : 1.0
                 wasting = True              neg : pos    =      7.6 : 1.0
                    scum = True              pos : neg    =      7.0 : 1.0
                  quaint = True              pos : neg    =      7.0 : 1.0
             wonderfully = True              pos : neg    =      6.8 : 1.0
              foreboding = True              pos : neg    =      6.4 : 1.0
                    sans = True              neg : pos    =      6.3 : 1.0
              mediocrity = True              neg : pos    =      6.3 : 1.0
             overwhelmed = True              pos : neg    =      5.7 : 1.0
                flawless = True              pos : neg    =      5.6 : 1.0
                   stark = True              pos : neg    =      5.4 : 1.0
              unoriginal = True              neg : pos    =      5.4 : 1.0
                  wasted = True              neg : pos    =      5.2 : 1.0
                   sunny = True              pos : neg    =      5.0 : 1.0
                searches = True              pos : neg    =      5.0 : 1.0
                   lofty = True              pos : neg    =      5.0 : 1.0
                 deadpan = True              pos : neg    =      5.0 : 1.0
                viewings = True              pos : neg    =      5.0 : 1.0
'''