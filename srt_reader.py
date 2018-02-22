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
        if tmp_sent.find(".") == -1:
          if i <= len(cf):
            cap = cf.get(i)
            tmp_sent += cap.txt + " "
            i += 1
            counter += 1
            prev_start = cap.start
          else:
            sent_cap.append(tmp_sent)
        else:
          split_ending = tmp_sent.split(".")
          ending = split_ending[0] + "."
          tmp_sent = '.'.join(split_ending[1:])
          i -= 1
          if start == 0:
            time_tuple = (prev_start, cap.end)
          else:
            time_tuple = (start, cap.end)
          if counter == 0:
            start = prev_start
          else:
            start = cap.start
          sent_cap.append(time_tuple)
          sent_cap.append(ending)
      print(i, sent_cap[0][0], sent_cap[0][1], sent_cap[1])
      i += 1


if __name__ == '__main__':
  caption_file = 'citynews_caption.srt'
  transcript_file = 'citynews_transcript.srt'
  
  with open(transcript_file) as tf:
    lines = tf.readlines()
    tcf = CaptionCollection(lines)
    ps = ParseSentence(tcf)
  
  
  '''
  with open(caption_file) as cf:
    lines = cf.readlines()
    ccf = CaptionCollection(lines)
    ps = ParseSentence(ccf)
  '''
  
  