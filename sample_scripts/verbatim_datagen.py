import csv
import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
from random import randint
import math

DATASIZE = 100000
portion = math.ceil(DATASIZE*0.2)


c = np.column_stack((r_sentence_sim, r_missing_words))
c = np.column_stack((c, pf_factors))

np.random.shuffle(c) # shuffle the order in rows

###### Simulated scores based on the fact generated from previous. ######
rating_list = []
# [delay], [speed], [verbatim factor score], [spelling and grammar error score], [missing words score] 
for i in c:
  verbatim_score, mw_score = 0,0
  # calculate delay rating
  sentence_sim = i[0]
  missing_words = i[1]
  pf_factor = i[2]
  
  '''
  # Paraphrasing (verbatimness) score which audiences subjectively feel
  if sentence_sim == 100 and missing_words == 0:
      verbatim_score = 10
  elif 96 <= sentence_sim < 100:
    if missing_words <= 1:
      verbatim_score = randint(8,10)
    elif 1 < missing_words <= 2:
      verbatim_score = randint(6,8)
    elif 2 < missing_words <= 5:
      verbatim_score = randint(4,7)
    else:
      verbatim_score = randint(0,3)
  elif 90 <= sentence_sim < 96: # over 95%
    if 0 < missing_words <= 2:
      verbatim_score = randint(5,8)
    elif 2 < missing_words <= 5:
      verbatim_score = randint(2,4)
    else:
      verbatim_score = randint(0,3)
  else:
    verbatim_score = randint(0,3)
  '''

  # Paraphrasing (verbatimness) score which audiences subjectively feel
  if sentence_sim == 100 and missing_words == 0:
      verbatim_score = 5
  elif 96 <= sentence_sim < 100:
    if missing_words <= 1:
      verbatim_score = 5
    elif 1 < missing_words <= 2:
      verbatim_score = 4
    elif 2 < missing_words <= 5:
      verbatim_score = 3
    else:
      verbatim_score = 1
  elif 90 <= sentence_sim < 96: # over 95%
    if 0 < missing_words <= 2:
      verbatim_score = 3
    elif 2 < missing_words <= 5:
      verbatim_score = 2
    else:
      verbatim_score = 1
  else:
    verbatim_score = 1
  
  #print(sentence_sim, missing_words, pf_factor, verbatim_score)

  vs = verbatim_score
  tmp_list = []
  for i in range(10):
    if i == vs-1:
      tmp_list.append(1)
    else:
      tmp_list.append(0)
  tmp_list.append(vs)
  rating_list.append(tmp_list)
  

p = np.asarray(rating_list)
c = np.column_stack((c, p))

print(c.shape) # For a matrix with n rows and m columns, shape will be (n,m)

filename = 'verbatimness_gen_dt_' + str(DATASIZE) + '.csv'
with open(filename, 'w') as mf:
  wr = csv.writer(mf)
  for i in c:
    wr.writerow(i)
