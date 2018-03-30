#srt file reader-parser
import re, sys, string
from caption import Caption
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as sw
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

import spacy
from spacy.lemmatizer import Lemmatizer

import numpy as np

#import enchant
#from enchant.checker import SpellChecker
'''
from keras.models import Sequential
from keras.layers import Dense

from sklearn import preprocessing
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
'''

class CaptionCollection:
  def __init__(self, lines):
    self.captionFile = {}
    self.makeCaption(lines)

  def put(self, key, val):
    self.captionFile[key] = val

  def get(self,key):
    return self.captionFile[key]

  def __str__(self):
    for k in self.captionFile:
      print(self.captionFile[k])
    return ""

  def __iter__(self):
    for k in self.captionFile:
      yield k

  def __len__(self):
    return len(self.captionFile)

  def is_sequence(self, line):
    return line.isnumeric()

  def is_time_stamp(self, line):
    return (line[:2].isnumeric() and line[2] == ':')

  def has_text(self, line):
    return re.search('[a-zA-Z]', line)

  def makeCaption(self, lines):
    i = 0
    line = str(lines[i]).strip()
    while i < len(lines):
      if len(line) > 0:
        if self.is_sequence(line):
          c = Caption()
          seq = int(line)
          c.setSeq(seq)
          i+=1 #nextline must be times
          line = str(lines[i]).strip()
          if self.is_time_stamp(line):
            c.setTime(line)
            i+=1 #nextline must be texts
            line = str(lines[i]).strip()
            textblock = ""
            while self.has_text(line):
              textblock += line + " " # should there be newline (\n) for multi-line captions
              i+=1 #check for next line
              if i<len(lines):
                line = str(lines[i]).strip()
              else:
                break
            c.setText(textblock.strip())
            self.captionFile[seq] = c
      else:
        i+=1 # next line.
        line = str(lines[i]).strip()

class Sentence:
  def __init__(self):
    self.time = []
    self.sentence = ""

  def setSentence(self, time_tuple, txt):
    self.time = tuple(time_list)
    self.sentence = txt

class ParseSentence:
  '''
    parsing the captions so that given a start-end time frame,
    - text chunks will be merged to be sentences.
    - then start to end timeframe will be used.

    -> then the sentences will be used to compare the meaning/paraphrasing...
  '''

  def __init__(self,cf):
    self.parseSentence = []
    self.parse(cf)

  def __str__(self):
    for c in cf:
      print(cf.get(c))
    return ""

  def __iter__(self):
    for k in self.parseSentence:
      yield k

  def __len__(self):
    return len(self.parseSentence)

  def get_sentences(self):
    return self.parseSentence

  def parse(self,cf):
    '''
      from a collection of captions (cf),
      iterate over each caption object,
      create a tuple of (start_time, end_time, sentence)
      such that the tuple includes,
        - sentence created from single/multiple caption objects
        - start time of the very first (start of sentence) caption object,
        - end time of the very last (end of sentence) caption object.
    '''
    i = 1
    tmp_sent = ""
    start, prev_start, end = 0, 0, 0

    while i <= len(cf):
      sent_cap = []
      counter = 0
      while not sent_cap:
        if tmp_sent.find(".") == -1 and tmp_sent.find("!") == -1 and tmp_sent.find("?") == -1:
          if i <= len(cf):
            cap = cf.get(i)
            tmp_sent += cap.txt + " "
            i += 1
            counter += 1
            prev_start = cap.start
          else:

            if start == 0:
              time_tuple = (prev_start, cap.end)
            else:
              time_tuple = (start, cap.end)
            sent_cap.append(time_tuple)
            sent_cap.append(tmp_sent.lower().strip())
        else:
          split_ending = re.split("[.!?]", tmp_sent)
          #split_ending = tmp_sent.split(".")
          ending = split_ending[0] + "."
          tmp_sent = '.'.join(split_ending[1:])
          i -= 1

          if start == 0:
            time_tuple = (prev_start, cap.end)
          else:
            time_tuple = (start, cap.end)
          sent_cap.append(time_tuple)
          final_sentence = ' '.join(handleOmissions(ending.lower().strip()))
          sent_cap.append(final_sentence)

          if counter == 0:
            start = prev_start
          else:
            start = cap.start
      #print(i, sent_cap[0][0], sent_cap[0][1], sent_cap[1])
      self.parseSentence.append(sent_cap)
      i += 1

def getSpellErr(s):
  exception_words = ["monday", "tuesday", "wednesday", "kilometres", 
                "simcoe", "york", "unionville", "markham", "ajax", 
                "whitby", "oakville", "november", "hamilton",
                "gta", "denise", "andreacchi", "niagara", "timmins",
                "bruce", "ctv", "paul",
                "santa", "claus", "grey", "ottawa", "ontario"]
  chkr = SpellChecker("en_US")
  #Check how many words in s1 have spelling errors
  chkr.set_text(s)
  counter = 0 # checking how many spelling errors in s1
  for err in chkr:
    if err.word not in exception_words:
      print("spell error at:", err.word)
      counter+=1
  return counter

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

def getSimilarity(s1, s2):
  c1_spacy = nlp(str(s1))
  c2_spacy = nlp(str(s2))

  #print(c1_spacy)
  #print(c2_spacy)

  c1_words, c2_words = {}, {}
  c1_ants, c2_ants = {}, {}

  for token in c1_spacy:
    if token.pos_=='VERB' or token.pos_=='ADJ' or token.pos_=='ADV':
      wordnet_token = wn.synsets(token.text)
      if wordnet_token: # if its in wornet db
        c1_words[token.text] = wordnet_token # add to the list

  for token in c2_spacy:
    if token.pos_=='VERB' or token.pos_=='ADJ' or token.pos_=='ADV':
      wordnet_token = wn.synsets(token.text)
      if wordnet_token: # if its in wornet db
        c2_words[token.text] = wordnet_token # add to the list

  # PRE: two dictionaries with word: word-synsets
  # POST: check whether there exists any antonyms in another sentence
  for w in c1_words.values(): # for each synset
    for s in w: # for each synonym
      for l in s.lemmas(): # find its lemma(s)
        if l.antonyms(): # check if it has an antonym or not
          for ants in l.antonyms(): # if there's antonym with this lemma
            c1_ants[ants.name()] = s # append into the ant dictionary

  for w in c2_words.values(): # for each synset
    for s in w: # for each synonym
      for l in s.lemmas(): # find its lemma(s)
        if l.antonyms(): # check if it has an antonym or not
          for ants in l.antonyms(): # if there's antonym with this lemma
            c2_ants[ants.name()] = s # append into the ant dictionary

  similarity_value = 0
  for w in c1_words.values():
    for s in w:
      word = s.name().split(".")[0]
      try:
        #print(s, c2_ants[word])
        similarity_value = s.wup_similarity(c2_ants[word])
      except:
        pass

  if similarity_value == 0:
    similarity_value = c1_spacy.similarity(c2_spacy)
  if similarity_value == None:
    similarity_value = 0

  return similarity_value

def getMissingWords(nlp, caption_s, transcript_s):
  # check whether caption sentence is missing words
  # from transcript sentence. which means, transcript
  # sentence will have more words than caption words
  # 
  missing = []
  table = "".maketrans("","",string.punctuation)
  a = caption_s.split(" ")
  b = transcript_s.split(" ")
  a_prime = []
  b_prime = []

  for i in a:
    word = i.translate(table)
    token = nlp(word)    
    if token:
      token = token[0]
      if token.lemma_ == "-PRON-":
        a_prime.append(word)
      else:
        a_prime.append(str(token.lemma_))
    else:
      a_prime.append(word)

  for i in b:
    word = i.translate(table)
    token = nlp(word)
    if token:
      token = token[0]
      if token.lemma_ == "-PRON-":
        b_prime.append(word)
      else:
        b_prime.append(str(token.lemma_))
    else:
      b_prime.append(word)

  # for each word in transcript,
  # check whether the word is in the caption sentence.
  for j in b_prime:
    if j not in a_prime:
      missing.append(j)

  #print(len(missing), missing)
  return len(missing), abs(len(a_prime)-len(b_prime))

def addValues(t_time,c_time,c_txt,t_txt,c_sentence,nlp):
  # calculate delay from the very first caption
  delay = abs(t_time[0].to_ms() - c_time[0].to_ms())
  # get words per min
  duration = (t_time[1].to_ms() - t_time[0].to_ms())/1000/60.0 # in minutes
  wpm = len(c_txt.split())/duration
  # similarity (paraphrasing)
  sim_value = getSimilarity(c_sentence[1], t_txt)
  # spelling errors
  #spelling = getSpellErr(c_sentence[1])
  # missing words?
  mw, wd = getMissingWords(nlp, c_sentence[1], t_txt)
  # append them all
  return [delay, wpm, sim_value, spelling, mw, wd, c_txt, t_txt]


if __name__ == '__main__':
  #caption_file = 'captions/citynews_caption.srt'
  #transcript_file = 'captions/citynews_transcript.srt'
  caption_file = 'captions/CTVnews_caption.srt'
  transcript_file = 'captions/CTVnews_transcript.srt'
  nlp = spacy.load('en')

  with open(caption_file) as cf:
      lines = cf.readlines()
      ccf = CaptionCollection(lines)
      cps = ParseSentence(ccf).get_sentences()

  with open(transcript_file) as tf:
    lines = tf.readlines()
    tcf = CaptionCollection(lines)
    tps = ParseSentence(tcf).get_sentences()
  
  # 1. value generation
  # a. find the delay first.
  input_matrix = []
  sync_delay = 0
  c_index, t_index = 0, 0

  #typically cps would have less sentences
  while c_index < len(cps):
    c_sentence = cps[c_index]
    c_time, c_txt = c_sentence[0], c_sentence[1]
    c_txt_ngram = ' '.join(c_txt.split()[0:2])

    t_sentence = tps[t_index]
    t_time, t_txt = t_sentence[0], t_sentence[1]
    t_txt_ngram = ' '.join(t_txt.split()[0:2])

    #check the first n-words, to see if it's the same sentence
    # this can be replaced to check similarity ..?
    if t_txt_ngram == c_txt_ngram:
      #print(t_txt_ngram, c_txt_ngram)
      break
    c_index += 1
  print("sync delay is then:", c_index)

  c_index = sync_delay # sync the delayed indices
  last_match_index = 0
  left_sentences = {}

  while t_index < len(tps):
    if c_index == len(cps):
      break
    c_sentence = cps[c_index]
    c_time, c_txt = c_sentence[0], c_sentence[1]
    c_txt_ngram = ' '.join(c_txt.split()[0:2])

    t_sentence = tps[t_index]
    t_time, t_txt = t_sentence[0], t_sentence[1]
    t_txt_ngram = ' '.join(t_txt.split()[0:2])

    c_txt_end_ngram = ' '.join(c_txt.split()[-3:])
    t_txt_end_ngram =  ' '.join(t_txt.split()[-3:])

    delay, duration, wpm, sim_value, spelling = 0,0,0,0,0

    if t_txt_ngram == c_txt_ngram:
      v_list = addValues(t_time,c_time,c_txt,t_txt,c_sentence,nlp)
      input_matrix.append(v_list)
      c_index += 1
      last_match_index = t_index
    elif t_txt_end_ngram == c_txt_end_ngram:
      v_list = addValues(t_time,c_time,c_txt,t_txt,c_sentence,nlp)
      input_matrix.append(v_list)
      c_index += 1
      last_match_index = t_index
    else:
      #print("no match, move the transcript sentence on..")
      left_sentences[c_time[0]] = c_sentence

    t_index += 1

    if t_index == len(tps) and c_index < len(cps)-1:
      #print("no match, move the Caption on...")
      c_index += 1
      t_index = last_match_index-1
  
  #print(len(cps), len(tps))
  
  #print(left_sentences)
  
  input_arr = []
  for i in input_matrix:
    tmp_arr = []
    tmp_arr.append([i[0]]) # delay
    tmp_arr.append([i[1]]) # wpm
    tmp_arr.append([i[2]]) # similarity
    tmp_arr.append([i[3]]) # sge
    tmp_arr.append([i[4]]) # mw
    tmp_arr.append([i[5]]) # md

    input_arr.append(tmp_arr)

  input_arr = np.asarray(input_arr)

  print("Citytv news: delay, wpm, sim_value, spelling, mw, wd,")
  print(np.mean(input_arr[:,0]), np.std(input_arr[:,0]),
        np.percentile(input_arr[:,0], [25,50,75]))
  print(np.mean(input_arr[:,1]), np.std(input_arr[:,1]),
        np.percentile(input_arr[:,1], [25,50,75]))
  print(np.mean(input_arr[:,2]), np.std(input_arr[:,2]),
        np.percentile(input_arr[:,2], [25,50,75]))
  print(np.mean(input_arr[:,3]), np.std(input_arr[:,3]),
        np.percentile(input_arr[:,3], [25,50,75]))
  print(np.mean(input_arr[:,4]), np.std(input_arr[:,4]),
        np.percentile(input_arr[:,4], [25,50,75]))
  print(np.mean(input_arr[:,5]), np.std(input_arr[:,5]),
        np.percentile(input_arr[:,5], [25,50,75]))


"""
Citytv news: delay, wpm, sim_value, spelling, mw, wd,
4895.75 1477.93 [7271 2669] [4075 4669.5 5775]
232.03 200.48 [814.159 57.143] [118.56 143.21 313.46]
0.854 0.1045 [1 0.689] [0.7770  0.8416  0.9467]
0.0 0.0 [ 0.] [ 0.]
4.833 6.243 [20 0] [0.75  1.5   7]
12.333 11.2496 [39 0] [1.75 11 20]

Citytv news: delay, wpm, sim_value, spelling, mw, wd,
7992.78947368 1888.85787209 [ 11910.] [ 3319.]
132.197394492 77.1592149465 [ 355.16664522] [ 48.37409299]
0.816991536733 0.289766435418 [ 1.00000002] [ 0.]
0.0 0.0 [ 0.] [ 0.]
5.21052631579 7.35261462531 [ 24.] [ 0.]
8.84210526316 11.0132129157 [ 34.] [ 0.]
"""