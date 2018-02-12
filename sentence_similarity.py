# for python 3
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as sw
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

#import spacy
#from spacy.lemmatizer import Lemmatizer

#from keras.preprocessing.text import text_to_word_sequence
#from keras.preprocessing.text import one_hot
#from keras.preprocessing.text import Tokenizer

import numpy as np
import re

#import enchant
#from enchant.checker import SpellChecker



from srt_reader import CaptionCollection

# Lemmatize/replace the words ommissions using apostrophe
# it's, let's, don't, doesn't, can't, you're, you aren't
# pre: str sentence
# post: list of words tokenized
def handleOmissions(sentence):
  s_list = sentence.split()
  i = 0
  while i < len(s_list):
    word = s_list[i]
    if word.lower() == "gonna":
      s_list.pop(i)
      s_list.insert(i, "going")
      s_list.insert(i+1, "to")
    if re.search("'[a-z]+", word): # when the word has apostrophe omission
      om_word = word.split("'")
      s_list.pop(i)
      if om_word[0].lower() == "let":
        s_list.insert(i, om_word[0])
        s_list.insert(i+1, "us")
      elif om_word[1].lower() == "s":
        s_list.insert(i, om_word[0])
        s_list.insert(i+1, "is")
      elif om_word[1].lower() == "t":
        # handle ain't and won't
        tmp_verb = om_word[0][:-1] # get rid of 'n'
        if tmp_verb.lower() == "ai": # aint
          tmp_verb = "am"
        elif tmp_verb.lower() == "wo": #wont
          tmp_verb = "will"
        else: # otherwise, do should would could did
          pass
        s_list.insert(i, tmp_verb)
        s_list.insert(i+1, "not")
      elif om_word[1].lower() == "re":
        s_list.insert(i, om_word[0])
        s_list.insert(i+1, "are")
      else:
        pass
    i+=1
  return s_list

# Given two sentences having different number of words
# find the absolute difference in number of words
def getWordDifference(s1, s2):
  return abs(len(s1)-len(s2))

def getSpellErr(s):
  chkr = SpellChecker("en_US")
  #Check how many words in s1 have spelling errors
  chkr.set_text(s)
  counter = 0 # checking how many spelling errors in s1
  for err in chkr:
    #print(err.word)
    counter+=1
  return counter

def checkSpeakerID(s):
  # check the first word
  # if it's a speaker id, 
  # it should have a semicolon at the end of the word.
  s_list = s.split()
  if re.search("[a-zA-Z]+:", s_list[0]):
    return True
  else:
    return False

#print('good', 'not good', nlp(str('good')).similarity(nlp(str('not good'))))
#print('good', 'bad', nlp(str('good')).similarity(nlp(str('bad'))))
#print('good', 'bad', wn.synsets('good')[0].wup_similarity(wn.synsets('bad')[0]))
def getSimilarity(s1, s2):
  # sentence comparison using SpaCy 101
  # spacy 101 uses n-gram language model,
  # based on wordvec uh
  #print(s1, s2, nlp(str(s1)).similarity(nlp(str(s2))))

  s1_words, s2_words = {}, {}
  s1_ants, s2_ants = {}, {}

  # for each word token in sentence 1
  for token in s1_spacy:
    # if the word is either a verb or an adjective
    if token.pos_=='VERB' or token.pos_=='ADJ' or token.pos_=='ADV':
      # check wordnet for its existence
      wordnet_token = wn.synsets(token.text)
      if wordnet_token: # if its in wornet db
        s1_words[token.text] = wordnet_token # add to the list

  # for each word token in sentence 2
  for token in s2_spacy:
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

def get_delay_from_captions(c1,c2):
  return 0

def get_speed_of_caption(c):
  return 0




if __name__ == '__main__':





  nlp = spacy.load('en')
  #nlp = spacy.load('en_core_web_lg')

  c1 = "and az you jst heard in sportz it's snowing in Ottawa"
  c1_spacy = nlp(str(c1))

  c2 = "Forecaster: AND Z YOU HEARD IN SPORTS IT IS SNOWING IN OTTAWA"
  c2_spacy = nlp(str(c2))

  # replace the omitted words
  # then, replace the sentences to the newly replaced sentences.
  c1_list = handleOmissions(c1)
  c2_list = handleOmissions(c2)
  c1 = ' '.join(c1_list)
  c2 = ' '.join(c2_list)

  #n1: missing words (count, which words)
  if len(c1) != len(c2):
    diff_wordcount = getWordDifference(c1,c2)
    print("number of words missing:", diff_wordcount)

  #n2: spelling/grammar errors? (check wordnet for existence?)
  #c1_typos = getSpellErr(c1)
  #c2_typos = getSpellErr(c2)
  #print(s1_typos, s2_typos)

  #n3: speaker ID (regex)
  '''
  A double chevron (>>) must be used to indicate each new speaker. 
  When the name of the speaker is known, 
    it must be included in mixed case followed by a colon. 

  Guests and others are designated using first and last names.
  People commonly associated with a broadcast will be designated 
  using first names only.

  In many instances, 
  graphics containing speaker identifications are covered by real-time captioning. 
  Caption stenographers must insert these identifications
  into the captions depending on the speed and complexity of the broadcast.
  '''
  s1_spID = checkSpeakerID(c1)
  s2_spID = checkSpeakerID(c2)

  #n4: delay of captions
  #      (delay-> 
  #       find the matching n-gram words-> 
  #       compare the start time)
  '''
  For live programming, the lag time between the audio and the
  captions must not exceed six seconds, averaged over the program.
  '''
  caption_file = 'citynews_caption.srt'
  
  with open(caption_file) as cf:
    lines = cf.readlines()
    ccf = CaptionCollection(lines)
    print(ccf)
    

  #print(get_delay_from_captions)

  # speed, word per minute.
  '''
  - Real-time captions appear in a three-line roll-up format at the bottom of the
  screen. This allows the maximum presentation rate for ease of reading by
  viewers.
  - However, in many cases, to avoid covering graphics, keys, and other essential
  visual information, captions will have to be moved to another location and
  displayed in a two-line roll-up. When captions change positions, the last
  caption in a segment must have at least a two-second duration before
  blanking and changing position.
  - Every new sentence and every new speaker should begin on a new line for
  ease of reading.
  - Captions must be blanked off the screen prior to entering a television
  commercial frame, and the encoder must be put in pass mode so as not to
  interfere with post-production commercial captions.

  -  Captions are limited to 32 characters per line of captions.
  '''
  #print(get_speed_of_caption())


  #n5: paraphrasing?
  #     a. tokenize, lemmatize the words ?
  #     b. Use SpaCy similarity first, ? (step a or b)
  #     c. Check antonyms, modify the similarity value



  #s1 = "it was rejected by the senate."
  #s2 = "it was accepted by the senate."

  #s1 = "i was satisfied."
  #s2 = "i was dissatisfied."

  #s1 = "IT'S ONLY GOING TO TAPER OFF FROM THIS POINT FORWARD."
  #s2 = "it's only going to taper off from this point forward and again it's about two to four maybe up to five centimeters of snow."

  # antonym used
  #s1 = "The quick brown fox jumps over the lazy dog."
  #s2 = "The quick brown fox didn't jump over the lazy dog."

  # adverb antonym..?
  #s1 = "The quick brown fox always jumps over the lazy dog."
  #s2 = "The quick brown fox never jump over the lazy dog."
