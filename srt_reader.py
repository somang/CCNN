#srt file reader-parser
import re, sys
from caption import Caption

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
              textblock += line + "\n" # should there be newline (\n) for multi-line captions
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
    while i <= len(cf): # for each caption object
      sent_cap = [] # create an empty list
      while not sent_cap: # while the list is empty
        counter = 0 # counter to check how many captions were connected.
        if tmp_sent.find(".") == -1:         # not the ending block
          cap = cf.get(i)                    # get the caption object
          tmp_sent += cap.txt + " "          # concatenate the sentence
          counter += 1       # increase the counter
          i += 1             # move on to next caption object
        else:                                # sentence ending block
          split_ending = tmp_sent.split(".") # split the sentence for carry on
          ending = split_ending[0] + "."
          tmp_sent = '.'.join(split_ending[1:]) # heading of another sentence
          if counter == 0:  # handling case of multiple sentences 
            i -= 1          # Don't move to next, since there are more sentences.
          sent_cap.append(ending) # pickup the completed sentence.
      print(i, sent_cap)
      i += 1 # move on.
    #while loop ends


if __name__ == '__main__':
  caption_file = 'citynews_caption.srt'
  transcript_file = 'citynews_transcript.srt'
  
  with open(transcript_file) as tf:
    lines = tf.readlines()
    cf = CaptionCollection(lines)
    ps = ParseSentence(cf)
    
  
  '''
  with open(caption_file) as cf:
    lines = cf.readlines()
    cf = CaptionCollection(lines)
    ps = ParseSentence()
    ps.parse(cf)
  '''

  
  