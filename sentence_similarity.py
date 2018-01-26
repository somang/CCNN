# for python 2
#!/usr/bin/python2.7


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as sw
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import spacy
from spacy.lemmatizer import Lemmatizer
#from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

import numpy

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer

nlp = spacy.load('en')
#nlp = spacy.load('en_vectors_web_lg')

# using apostrophe: should be handled separatly (hardcoded..?)..
# it's, let's, don't, doesn't, can't, etc.

#s1 = "it was rejected by the senate."
#s2 = "it was accepted by the senate."

s1 = "i was satisfied."
s2 = "i was dissatisfied."

#s1 = "IT'S ONLY GOING TO TAPER OFF FROM THIS POINT FORWARD."
#s2 = "it's only going to taper off from this point forward and again it's about two to four maybe up to five centimeters of snow."

# antonym used
#s1 = "The quick brown fox jumps over the lazy dog."
#s2 = "The quick brown fox didn't jump over the lazy dog."

# adverb antonym..?
#s1 = "The quick brown fox always jumps over the lazy dog."
#s2 = "The quick brown fox never jump over the lazy dog."

# week 2 Todo list.
#n1: missing words (count, which words)
#n2: spelling/grammar errors? (check wordnet for existence?)
#n3: speaker ID (regex)

#n4: speed of captions
#      (delay-> 
#       find the matching n-gram words-> 
#       compare the start time)

#n5: paraphrasing?
#     a. tokenize, lemmatize the words ?
#     b. Use SpaCy similarity first, ? (step a or b)
#     c. Check antonyms, modify the similarity value


#handling apostrophe use












# sentence comparison using SpaCy 101
# spacy 101 uses n-gram language model,
# based on wordvec uh
print(s1, s2, nlp(unicode(s1)).similarity(nlp(unicode(s2))))

s1_words, s2_words = {}, {}
s1_ants, s2_ants = {}, {}

# for each word token in sentence 1
for token in nlp(unicode(s1)):
  # if the word is either a verb or an adjective
  if token.pos_=='VERB' or token.pos_=='ADJ' or token.pos_=='ADV':
    # check wordnet for its existence
    '''
    if token.pos_=='VERB':
      wordnet_token = wn.synsets(token.text, pos=wn.VERB)
    elif token.pos_=='ADJ':
      wordnet_token = wn.synsets(token.text, pos=wn.ADJ)
    elif token.pos_=='ADV':
      wordnet_token = wn.synsets(token.text, pos=wn.ADV)
    '''
    wordnet_token = wn.synsets(token.text)
    if wordnet_token: # if its in wornet db
      s1_words[token.text] = wordnet_token # add to the list

# for each word token in sentence 2
for token in nlp(unicode(s2)):
  # if the word is either a verb or an adjective
  if token.pos_=='VERB' or token.pos_=='ADJ' or token.pos_=='ADV':
    # check wordnet for its existence
    '''
    if token.pos_=='VERB':
      wordnet_token = wn.synsets(token.text, pos=wn.VERB)
    elif token.pos_=='ADJ':
      wordnet_token = wn.synsets(token.text, pos=wn.ADJ)
    elif token.pos_=='ADV':
      wordnet_token = wn.synsets(token.text, pos=wn.ADV)
    '''
    wordnet_token = wn.synsets(token.text)
    if wordnet_token: # if its in wornet db
      s2_words[token.text] = wordnet_token # add to the list

#nlp = spacy.load('en_vectors_web_lg')

# PRE: two dictionaries with word: word-synsets
# POST: check whether there exists any antonyms in another sentence
for w in s1_words.values(): # for each synset
  for s in w: # for each synonym
    for l in s.lemmas(): # find its lemma(s)
      if l.antonyms(): # check if it has an antonym or not
        for ants in l.antonyms(): # if there's antonym with this lemma
          s1_ants[ants.name()] = s # append into the ant dictionary

for w in s2_words.values(): # for each synset
  for s in w: # for each synonym
    for l in s.lemmas(): # find its lemma(s)
      if l.antonyms(): # check if it has an antonym or not
        for ants in l.antonyms(): # if there's antonym with this lemma
          s2_ants[ants.name()] = s # append into the ant dictionary

#print(s2_ants)

for w in s1_words.values():
  for s in w:
    word = s.name().split(".")[0]
    try:
      print(s, s2_ants[word])
      print(s.wup_similarity(s2_ants[word]))
    except:
      pass

#print(wn.synsets('satisfied')[0].lemmas()[0].antonyms())
#print(wn.synsets('dissatisfied')[0].lemmas()[0].antonyms())





#print(w1.text, w1.pos_, w2.text ,w2.pos_, sim)

#word comparison
#print(nlp(u'satisfied').similarity(nlp(u'dissatisfied')))

# Hyponymy? red is hyponym of color (opposite is hypernym)
# Meronymy? finger is meronym of hand (opposite is holonym)
# hypernyms/hyponym


# It's important to note that the `Synsets` from such a query are ordered by 
# how frequent that sense appears in the corpus
 
# You can read more about the different types of wordnet similarity 
# measures here: http://www.nltk.org/howto/wordnet.html 

'''
tokens = nlp(u'dog cat feline mammal')
#for token1 in tokens:
#    for token2 in tokens:
#        print(token1.text, token2.text, token1.similarity(token2))
#for token in tokens:
#  print(token.text, token.vector_norm)
'''


'''
stop_words = set(sw.words("english"))
for w in text_to_word_sequence(s1):
  if w not in stop_words:
    unicode_w = nlp(unicode(w))[0].lemma_
    #print(unicode_w)
    filtered_s1.append(unicode_w)
'''



