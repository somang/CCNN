import csv
import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
from random import randint
import math

gen_data = []
#https://crtc.gc.ca/eng/archive/2012/2012-362.htm

DATASIZE = 100
portion = math.ceil(DATASIZE*0.1)

# delay, wpm, similarity, number of errors
r_delay = np.random.uniform(
  low=0.0, high=10000.0, size=(DATASIZE,1))
r_wpm = np.random.uniform(
  low=0.0, high=400.0, size=(DATASIZE,1))
r_sentence_sim = np.random.uniform(
  low=0.0, high=100.0, size=(DATASIZE-portion,1)) # sentence cosine similarity

r_spell_grammar_errors = np.random.randint(10, size=(DATASIZE,1)) # random number of spelling and grammar errors
r_missing_words = np.random.randint(10, size=(DATASIZE,1)) # random number of missing words

max_ss = []
for i in range(portion):
  max_ss.append([100.0])

max_ss = np.asarray(max_ss)
r_sentence_sim = np.concatenate((r_sentence_sim, max_ss))

# paraphrasing factor:
pf_factors = []
pf = 0
for i in range(DATASIZE):
  if i < 100:
    missing_word_count = r_missing_words[i]
    if missing_word_count > 0:
      # when there are missing words, 
      # then it means there has been paraphrasing
      pf = 1
    else:
      pf = 0
  else:
    pf = 0
  pf_factors.append(pf)
pf_factors = np.asarray(pf_factors)

c = np.column_stack((r_delay, r_wpm))
c = np.column_stack((c, r_sentence_sim))
c = np.column_stack((c, r_spell_grammar_errors))
c = np.column_stack((c, r_missing_words))
c = np.column_stack((c, pf_factors))

np.random.shuffle(c) # shuffle the order?


###### Simulated scores based on the fact generated from previous. ######
rating_list = [[],[],[],[],[]]
# [delay], [speed], [verbatim factor score], [spelling and grammar error score], [missing words score] 
for i in c:
  delay_score, speed_score, verbatim_score, sge_score, missing_words_score = 0,0,0,0,0
  # calculate delay rating
  delay = i[0]
  wpm = i[1]
  sentence_sim = i[2]
  spell_grammar_errors = i[3]
  missing_words = i[4]

  if delay <= 100:
    delay_score = 10
  elif 100 < delay <= 500:
    delay_score = randint(8, 9)
  elif 500 < delay <= 2000:
    delay_score = randint(4, 7)
  elif 2000 < delay <= 4000:
    delay_score = randint(2, 4)
  else:
    delay_score = randint(0, 3)
  
  # calculate speed_rating
  if wpm <= 90:
    speed_score = randint(0,4) # when its too slow to read
  elif 90 < wpm <= 100:
    speed_score = randint(5,8)
  elif 100 < wpm <= 120:
    speed_score = randint(8,10)
  elif 120 < wpm <= 140:
    speed_score = randint(9,10)
  elif 140 < wpm <= 220:
    speed_score = randint(5,8)
  else:
    speed_score = randint(0,4)

  # Paraphrasing (verbatimness) score which audiences subjectively feel
  if sentence_sim == 100:
      verbatim_score = 10
  elif 96 <= sentence_sim < 100:
    if missing_words > 0:
      verbatim_score = randint(8,9)
    elif 0 < missing_words < 2:
      verbatim_score = randint(5,8)
  elif 90 <= sentence_sim < 96: # over 95%
    if 0 < missing_words <= 2:
      verbatim_score = randint(4,7)
    elif 2 < missing_words <= 5:
      verbatim_score = randint(2,4)
    elif 5 < missing_words:
      verbatim_score = randint(0,3)
  else:
    verbatim_score = randint(0,3)
  
  if spell_grammar_errors == 0:
    sge_score = 10
  elif spell_grammar_errors <= randint(2,3):
    sge_score = randint(8,10)
  elif 2 < spell_grammar_errors <= 5:
    sge_score = randint(3,7)
  else:
    sge_score = randint(0,2)

  if missing_words == 0:
    missing_words_score = 10
  elif 0 < missing_words <= randint(2,3):
    missing_words_score = randint(5,9)
  elif 3 < missing_words <= 5:
    missing_words_score = randint(4,6)
  else:
    missing_words_score = randint(0,3)
  
  rating_list[0].append(delay_score)
  rating_list[1].append(speed_score)
  rating_list[2].append(verbatim_score)
  rating_list[3].append(sge_score)
  rating_list[4].append(missing_words_score)

p = np.asarray(rating_list)

for i in p:
  c = np.column_stack((c, i))

print(c)
print(c.shape) # For a matrix with n rows and m columns, shape will be (n,m)

filename = 'gen_dt_' + str(DATASIZE) + '.csv'
with open(filename, 'w') as mf:
  wr = csv.writer(mf)
  for i in c:
    wr.writerow(i)
