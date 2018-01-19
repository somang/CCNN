from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as sw
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import spacy
import numpy

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer

nlp = spacy.load('en')
#nlp = spacy.load('en_vectors_web_lg')

s1 = "A Cat is satisfied by university."
s2 = "A Cat is dissatisfied by university." #"Water is to feed the dog"
s3 = "In order to feed a dog, you need water"

filtered_s1 = []
filtered_s2 = []
filtered_s3 = []
'''
stop_words = set(sw.words("english"))
for w in text_to_word_sequence(s1):
  if w not in stop_words:
    unicode_w = nlp(unicode(w))[0].lemma_
    #print(unicode_w)
    filtered_s1.append(unicode_w)

for w in text_to_word_sequence(s2):
  if w not in stop_words:
    unicode_w = nlp(unicode(w))[0].lemma_
    #print(unicode_w)
    filtered_s2.append(unicode_w)
'''

for token in nlp(unicode(s1)):
    #print(token.lemma_, token.pos_, token.tag_, token.dep_)
    if token.pos_=='VERB' or token.pos_=='ADJ':
      filtered_s1.append(token)

for token in nlp(unicode(s2)):
    if token.pos_=='VERB' or token.pos_=='ADJ':
      filtered_s1.append(token)

for w1 in filtered_s1:
  word_one = wn.synsets(w1.text)[0]
  for lemma in word_one.lemmas():
    ant = lemma.antonyms()
    if ant:
      print(lemma, ant)

    #sim = wn.synsets(w1.text)[0].wup_similarity(wn.synsets(w2.text)[0])
    #print(w1.text, w1.pos_, w2.text ,w2.pos_, sim)

# sentence comparison
#print(s1, s2, nlp(unicode(s1)).similarity(nlp(unicode(s2))))
#print(s1, s3, nlp(unicode(s1)).similarity(nlp(unicode(s3))))
#print(s2, s3, nlp(unicode(s2)).similarity(nlp(unicode(s3))))

#word comparison
#nlp = spacy.load('en_core_web_lg')
#print(nlp(u'satisfied').similarity(nlp(u'dissatisfied')))


# word comparison using wordnet
#ac = wn.synsets('satisfied','v')[0]
#rj = wn.synsets('dissatisfied','v')[0]
#print(ac.wup_similarity(rj))

# using language model of nlp = spacy.load('en'), the sentence similarity
# seems to be accurate, while the word similarities are not.
# However, the spacy similarities do not encounter for the antonym uses,
# which can completly change the meaning of the sentence or intention.

# the wordnet synsets wup_similarity gives the better similarity score 
# in terms of word meaning comparison.




# Hyponymy? red is hyponym of color (opposite is hypernym)
# Meronymy? finger is meronym of hand (opposite is holonym)


# wordnet: lexical database for the english language
# 
# hypernyms/hyponym
# antonyms
# 











# It's important to note that the `Synsets` from such a query are ordered by 
# how frequent that sense appears in the corpus

# You can check out how frequent a lemma is by doing:
#cat = wn.synsets('cat', 'n')[0]     # Get the most common synset
#dog = wn.synsets('dog', 'n')[0]           # Get the most common synset
#feline = wn.synsets('feline', 'n')[0]     # Get the most common synset
#mammal = wn.synsets('mammal', 'n')[0]     # Get the most common synset
 
# You can read more about the different types of wordnet similarity 
# measures here: http://www.nltk.org/howto/wordnet.html
#for synset in [cat, dog, feline, mammal]:
#    print "Similarity(%s, %s) = %s" % (cat, synset, cat.wup_similarity(synset))
 
# Similarity(Synset('cat.n.01'), Synset('dog.n.01')) = 0.2
# Similarity(Synset('cat.n.01'), Synset('feline.n.01')) = 0.5
# Similarity(Synset('cat.n.01'), Synset('mammal.n.01')) = 0.2

'''
print('spacy below')
nlp = spacy.load('en_vectors_web_lg')
tokens = nlp(u'dog cat feline mammal')

#for token1 in tokens:
#    for token2 in tokens:
#        print(token1.text, token2.text, token1.similarity(token2))
#for token in tokens:
#  print(token.text, token.vector_norm)

doc0 = nlp(u"Paris is the smallest city in France.")
doc1 = nlp(u"Paris is the largest city in France.")
doc2 = nlp(u"Vilnius is the capital of Lithuania.")
doc3 = nlp(u"An emu is a large bird.")

for doc in [doc0, doc1, doc2, doc3]:
    for other_doc in [doc0, doc1, doc2, doc3]:
        print(doc, other_doc, doc.similarity(other_doc))
'''




