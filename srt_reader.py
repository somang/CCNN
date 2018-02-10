#srt file reader-parser
import re, sys
from caption import Caption

class CaptionFile:
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
              textblock += line + " "
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

  def __init__(self):
    self.parseSentence = []

  def __str__(self):
    for c in cf:
      print(cf.get(c))
    return ""

  def parse(self, cf):
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
    while i <= len(cf):
      sent_cap = []
      while not sent_cap:
        counter = 0
        if tmp_sent.find(".") == -1:         # not the ending block
          cap = cf.get(i)
          tmp_sent += cap.txt + " "
          counter += 1
          i += 1
        else:                                # ending block
          split_ending = tmp_sent.split(".")
          ending = split_ending[0] + "."
          tmp_sent = '.'.join(split_ending[1:])
          if counter == 0:
            i -= 1
          sent_cap.append(ending)
      #print(tmp_sent)
      print(i, sent_cap)
      i += 1
    #while loop ends







if __name__ == '__main__':
  #file_name = 'citynews_caption.srt'
  file_name = 'citynews_transcript.srt'
  
  with open(file_name) as f:
    lines = f.readlines()
    cf = CaptionFile(lines)
    ps = ParseSentence()
    ps.parse(cf)
    #print(ps)