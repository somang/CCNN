import numpy
import doctest
from sklearn.datasets import fetch_20newsgroups

def lvd(r, h):
  """
  #https://martin-thoma.com/word-error-rate-calculation/
  Calculation of Levenshtein distance.

  Works only for iterables up to 254 elements (uint8).
  O(nm) time ans space complexity.

  Splited words can be compared to calculate WER.

  Parameters
  ----------
  r : sentence
  h : another sentence

  Returns
  -------
  int

  Examples
  --------
  >>> lvd("who is there anyway".split(), "is there someone".split())
  2
  >>> lvd("who is there".split(), "is there".split())
  1
  >>> lvd("who is there".split(), "".split())
  3
  >>> lvd("".split(), "who is there".split())
  3
  >>> lvd("who is there anyway", "is there someone")
  11
  >>> lvd("who is there", "is there")
  4
  >>> lvd("who is there", "")
  12
  >>> lvd("", "who is there")
  12

  """
  # initialisation
  d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
  d = d.reshape((len(r)+1, len(h)+1))
  # padding with zeroes and index numbers
  for i in range(len(r)+1):
      for j in range(len(h)+1):
          if i == 0:
              d[0][j] = j
          elif j == 0:
              d[i][0] = i
  #print(d)

  # computation
  for i in range(1, len(r)+1):
    for j in range(1, len(h)+1):
      #print("Word:", r[i-1], h[j-1])

      if r[i-1] == h[j-1]: # find the match
        d[i][j] = d[i-1][j-1]
        #print(d)
      else:
        substitution = d[i-1][j-1] + 1
        insertion    = d[i][j-1] + 1
        deletion     = d[i-1][j] + 1
        #print(substitution, insertion, deletion)
        d[i][j] = min(substitution, insertion, deletion)
  return d[len(r)][len(h)]

if __name__ == "__main__":
  #doctest.testmod()
  caption = "who is there anyway".split()
  transcript = "is there someone".split()
  edit_distance = lvd(caption, transcript)
  print(edit_distance, len(transcript))
  print('wer: {0:2.2f}%'.format( edit_distance/len(transcript)*100) )


  twenty = fetch_20newsgroups()
