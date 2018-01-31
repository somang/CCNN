#srt file reader-parser
import re, sys

def is_time_stamp(line):
  if line[:2].isnumeric() and line[2] == ':':
    return True
  return False

def has_letters(line):
  if re.search('[a-zA-Z]', line):
    return True
  return False


def main():
  file_name = 'citynews_caption.srt'
  with open(file_name) as f:
    lines = f.readlines()
    for i in range(len(lines)):
      line = str(lines[i])
      print(line, is_time_stamp(line), has_letters(line))



if __name__ == '__main__':
  main()