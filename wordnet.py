# -*- coding: utf-8 -*-
"""
@author: Aaroh
"""

from nltk.corpus import wordnet

#List of all synonyms of word Program
syns = wordnet.synsets("Program")
#Display the whole list
print(syns)
#Display first element
print(syns[0])
#Display only the word
print(syns[0].name())
print(syns[0].lemmas())
print(syns[0].lemmas()[0].name())

#Definition of the word
print(syns[0].definition())

#Examples of the words
print(syns[0].examples())

#List of synonyms and antonyms of word good
syno = []
anto = []

syns = wordnet.synsets("good")
for i in syns:
    for j in i.lemmas():
        #Appending all sysnonyms
        syno.append(j.name())
        #Checking if an antonym exist
        if j.antonyms():
            #Appending all antonyms
            anto.append(j.antonyms()[0].name())

#Display Synonyms and Antonyms
print(syno)
print(anto)

#Synonym list of words
w1 = wordnet.synsets("Car")
w2 = wordnet.synsets("Truck")
w3 = wordnet.synsets("Tiger")
w4 = wordnet.synsets("Ant")

#Check how much their meanings are same
#It return value between 0 to 1
print(w1[0].wup_similarity(w2[0]))
print(w1[0].wup_similarity(w3[0]))
print(w1[0].wup_similarity(w4[0]))