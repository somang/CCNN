#srt file reader-parser
import re, sys
from caption import Caption

def is_sequence(line):
  #print(repr(line[-1])) # to print the escaped chunk
  if line.isnumeric():
    return True
  return False

def is_time_stamp(line):
  if line[:2].isnumeric() and line[2] == ':':
    return True
  return False

def has_text(line):
  if re.search('[a-zA-Z]', line):
    return True
  return False

def makeCaption(lines):
  i = 0
  while i < len(lines):
    line = str(lines[i]).strip()
    if len(line) > 0:
      c = Caption()
      if is_sequence(line):
        c.setSeq(int(line))
      elif is_time_stamp(line):
        c.setTime(line)
      else:
        c.setText(line)
    
    i+=1 # next line.


if __name__ == '__main__':
  #file_name = 'citynews_caption.srt'
  file_name = 'citynews_transcript.srt'
  
  with open(file_name) as f:
    lines = f.readlines()
    makeCaption(lines)
