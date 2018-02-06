#srt file reader-parser
import re, sys
from caption import Caption

class CaptionFile:
  def __init__(self):
    self.captionFile = {}

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


class ParseSentence:
  '''
    parsing the captions so that given a start-end time frame,
    - text chunks will be merged to be sentences.
    - then start to end timeframe will be used.

    -> then the sentences will be used to compare the meaning/paraphrasing...
  '''

  def __init__(self):
    self.parseSentence = []

  def parse(self, cf):
    for c in cf:
      print(cf.get(c))
  
  def __str__(self):
    for c in cf:
      print(cf.get(c))
    return ""






def is_sequence(line):
  return line.isnumeric()

def is_time_stamp(line):
  return (line[:2].isnumeric() and line[2] == ':')

def has_text(line):
  return re.search('[a-zA-Z]', line)

def makeCaption(lines):
  captionFile = CaptionFile()
  i = 0
  line = str(lines[i]).strip()
  while i < len(lines):
    if len(line) > 0:
      if is_sequence(line):
        c = Caption()
        seq = int(line)
        c.setSeq(seq)
        i+=1 #nextline must be times
        line = str(lines[i]).strip()
        if is_time_stamp(line):
          c.setTime(line)
          i+=1 #nextline must be texts
          line = str(lines[i]).strip()
          textblock = ""
          while has_text(line):
            textblock += line + " "
            i+=1 #check for next line
            if i<len(lines):
              line = str(lines[i]).strip()
            else:
              break
          c.setText(textblock)
          captionFile.put(seq,c)
    else:
      i+=1 # next line.
      line = str(lines[i]).strip()
  return captionFile

if __name__ == '__main__':
  file_name = 'citynews_caption.srt'
  #file_name = 'citynews_transcript.srt'
  
  with open(file_name) as f:
    lines = f.readlines()
    cf = makeCaption(lines)
    #print(cf)
    ps = ParseSentence()
    ps.parse(cf)
    #print(ps)