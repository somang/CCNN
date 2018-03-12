#srt file reader-parser
import re, sys
from caption import Caption
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as sw
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

import spacy
from spacy.lemmatizer import Lemmatizer

import numpy as np

import enchant
from enchant.checker import SpellChecker

from keras.models import Sequential
from keras.layers import Dense


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



if __name__ == '__main__':
  caption_file = 'captions/citynews_caption.srt'
  transcript_file = 'captions/citynews_transcript.srt'
  #caption_file = 'CTVnews_caption.srt'
  #transcript_file = 'CTVnews_transcript.srt'

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

    #print(c_index, len(cps))
    #print(c_txt)

    if t_txt_ngram == c_txt_ngram:
      #print("c:", c_txt)
      #print("t:", t_txt)
      #print()

      # calculate delay from the very first caption
      delay = abs(t_time[0].to_ms() - c_time[0].to_ms())
      # get words per min
      duration = (t_time[1].to_ms() - t_time[0].to_ms())/1000/60.0 # in minutes
      wpm = len(c_txt.split())/duration
      # similarity (paraphrasing)
      sim_value = getSimilarity(c_sentence[1], t_txt)
      # spelling errors
      spelling = getSpellErr(c_sentence[1])
      # append them all
      v_list = [delay, wpm, sim_value, spelling, c_txt, t_txt]
      input_matrix.append(v_list)
      c_index += 1
      last_match_index = t_index
    elif t_txt_end_ngram == c_txt_end_ngram:
      #print("ENDING MATCH")
      #print("c:", c_txt)
      #print("t:", t_txt)
      #print()
      # calculate delay from the very first caption
      delay = abs(t_time[0].to_ms() - c_time[0].to_ms())
      # get words per min
      duration = (t_time[1].to_ms() - t_time[0].to_ms())/1000/60.0 # in minutes
      wpm = len(c_txt.split())/duration
      # similarity (paraphrasing)
      sim_value = getSimilarity(c_sentence[1], t_txt)
      # spelling errors
      spelling = getSpellErr(c_sentence[1])
      # append them all
      #print(wpm, c_txt, duration, len(c_txt.split()))
      v_list = [delay, wpm, sim_value, spelling, c_txt, t_txt]
      input_matrix.append(v_list)
      c_index += 1
      last_match_index = t_index
    else:
      #print("no match, move the transcript sentence on..")
      #print()
      left_sentences[c_time[0]] = c_sentence

    t_index += 1

    if t_index == len(tps) and c_index < len(cps)-1:
      #print("no match, move the Caption on...")
      c_index += 1
      t_index = last_match_index-1
  
  #print(len(cps), len(tps))
  #print(input_matrix, len(input_matrix))
  #print(left_sentences)

  # setup data
  #dataset = np.loadtxt("mixed_data.csv", delimiter=",")
  dataset = np.genfromtxt("mixed_data.csv", delimiter=",")

  # split input(X) and output(Y)
  X = []
  Y = dataset[:,:]
  dY = []
  dX = []

  for i in range(len(input_matrix)):
    input_matrix[i].append(Y[:,0][i])
    print(input_matrix[i])
    X.append(input_matrix[i])

  X = np.asarray(X)
  Y = Y[:12,1:7] # 0 = D, 1 = HOH, 2 = H
  #print(X)
  #print(Y)

  '''
  
  #create model
  model = Sequential()
  model.add(Dense(32, input_dim=5, activation='relu'))
  model.add(Dense(12, activation='relu'))
  model.add(Dense(6, kernel_initializer='normal'))

  #compile model
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

  #fit the model
  history = model.fit(X, Y, epochs=100, batch_size=10)

  #predict using the model
  p_input = np.array(
    [
      [100, 548.1481481481483, 1.0000000001558595, 0, 0], 
      [6820, 582.0467276950403, 0, 0, 1]
    ]
    )
  #print(p_input.shape)

  prediction = model.predict(p_input)
  print(prediction)
  '''
  # the order of input goes
  # delay
  # word per minute
  # similarity_value
  # spelling


  # the order of prediction output goes
  # how much do you think the 
  # 0. fast apeearing and disappearing captions
  # 1. slow apeearing and disappearing captions
  # 2. missing words
  # 3. spelling errors
  # 4. speaker identification 
  # 5. verbatim accurate captions 
  # affect viewing pleasure?
  '''
  0,3319,548.1481481481483,0.94170733293952125,0,4,3,5,4,6,3
  0,6820,582.0467276950403,0.88214744364563769,0,7,5,4,3,0,1
  0,7690,866.9527896995708.96485043782697499,0,5,5,4,4,3,3
  0,6470,490.2506963788301,0,0.98002335523151629,0,3,5,2,3,2,4
  0,6670,772.3577235772358,0.98637051811050569,0,7,8,7,8,8,9
  0,9280,51.61567826227216,0.88214744364563769,0,3,2,4,4,3,4
  0,6441,225.74576726686374,1.0000000191147738,0,4,4,4,4,4,4
  0,6941,886.8583714055361,1.000000014566867,0,4,7,5,8,5,3
  0,8941,1251.7491413306195,0.84146567103695302,1,3,8,2,2,4,3
  0,9620,854.5205190420932,0.76122313894492211,0,2,5,3,5,10,3
  1,7541,1798.9962681765537,0.85008997448299106,0,
  1,8820,766.497461928934,0.93659734451794641,0,
  1,8810,171.42857142857144,0.81687708963985195,0,
  1,11910,260.586319218241,0.76841326916754171,0,
  1,8760,75.34246575342466,0.89039354114681157,0,
  1,6220,137.93103448275863,0.94874995546497143,0, 
  1,9200,446.6666666666667,0.95393009193162548,0,
  '''