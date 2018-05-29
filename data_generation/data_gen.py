import csv
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from scipy.stats import truncnorm
import math

#https://crtc.gc.ca/eng/archive/2012/2012-362.htm
def get_truncated_normal(mean=0, sd=0, low=0, high=10):
  value = truncnorm(
    (low - mean) / sd, 
    (high - mean) / sd, 
    loc=mean, scale=sd)
  return value

def score_normalization(x, range):
  if range == 10:
    return x
  elif range == 5:
    if 0 <= x <= 2: return 1
    elif 3 <= x <= 4: return 2
    elif 5 <= x <= 6: return 3
    elif 7 <= x <= 8: return 4
    elif 8 <= x <= 10: return 5
    else: return 0
  elif range == 3:
    if 0 <= x <= 3: return 1
    elif 4 <= x <= 6: return 2
    elif 7 <= x <= 10: return 3
    else: return 0

SCALE = 10
DATASIZE = 100000
#print("SCALE:",SCALE,", SIZE:",DATASIZE)
# delay, wpm, similarity, number of errors
### normal distribution using the mean and sd from existing data.
trn = get_truncated_normal(mean=4895.75, sd=1477.94, low=0, high=12000)
r_delay = trn.rvs(DATASIZE)

trn = get_truncated_normal(mean=232.03, sd=200.48, low=0, high=850)
r_wpm = trn.rvs(DATASIZE)

trn = get_truncated_normal(mean=1, sd=2, low=0.0, high=10)
r_spell_grammar_errors = np.rint(trn.rvs(DATASIZE))

trn = get_truncated_normal(mean=0.85, sd=0.2, low=0.0, high=1.01)
r_sentence_sim = trn.rvs(DATASIZE)

trn = get_truncated_normal(mean=5.02, sd=6.79, low=0.0, high=25)
r_missing_words = []
for i in range(DATASIZE):
  # because it cannot have a 100% and a missing word..
  if r_sentence_sim[i] > 1.0:
    r_sentence_sim[i] = 1.0
    r_missing_words.append(0)
  else:
    mw = np.rint(trn.rvs())
    r_missing_words.append(mw)
r_missing_words = np.asarray(r_missing_words)

# generate paraphrasing factor:
pf_factors = []
pf = 0
for i in range(DATASIZE):
  if r_sentence_sim[i] < 1.0:
    if r_missing_words[i] > 0:
      # when there are missing words, 
      # then it means there has been paraphrasing
      pf = 1
  pf_factors.append(pf)
  pf = 0
pf_factors = np.asarray(pf_factors)

c = np.column_stack((r_delay, r_wpm))
c = np.column_stack((c, r_spell_grammar_errors))
c = np.column_stack((c, r_missing_words))
c = np.column_stack((c, r_sentence_sim))
c = np.column_stack((c, pf_factors))

np.random.shuffle(c) # shuffle the order in rows

###### Simulated scores based on the fact generated from previous. ######
rating_list = [[],[],[],[],[]]
# [delay], [speed], [verbatim factor score], [spelling and grammar error score], [missing words score] 
for i in c:
  delay_score, speed_score, verbatim_score, sge_score, missing_words_score = 0,0,0,0,0
  # calculate delay rating
  delay = i[0]
  wpm = i[1]
  spell_grammar_errors = i[2]
  missing_words = i[3]
  sentence_sim = i[4]

  if delay <= 1000:
    delay_score = randint(8,10)
  elif 1000 < delay <= 2000:
    delay_score = randint(4, 7)
  elif 2000 < delay <= 4000:
    delay_score = randint(2, 4)
  else:
    delay_score = randint(0, 2)
  
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
  if sentence_sim == 1.0:
      verbatim_score = 10
  elif 0.96 <= sentence_sim < 1.0:
    if 0 < missing_words <= 2:
      verbatim_score = randint(8,9)
    elif 2 < missing_words < 5:
      verbatim_score = randint(5,8)
    elif 5 < missing_words < 10:
      verbatim_score = randint(3,6)
    else:
      verbatim_score = randint(0,3)
  elif 0.9 <= sentence_sim < 0.96: # over 95%
    if 0 < missing_words <= 2:
      verbatim_score = randint(6,8)
    elif 2 < missing_words < 5:
      verbatim_score = randint(5,8)
    elif 5 < missing_words < 10:
      verbatim_score = randint(3,5)
    else:
      verbatim_score = randint(0,3)
  else:
    verbatim_score = randint(0,2)
  
  tmp_flag = randint(2,3)
  if spell_grammar_errors == 0:
    sge_score = 10
  elif spell_grammar_errors <= tmp_flag:
    sge_score = randint(8,10)
  elif tmp_flag < spell_grammar_errors <= 5:
    sge_score = randint(4,7)
  else:
    sge_score = randint(0,3)

  if missing_words == 0:
    missing_words_score = 10
  elif 0 < missing_words <= randint(2,3):
    missing_words_score = randint(5,9)
  elif 3 < missing_words <= 5:
    missing_words_score = randint(4,7)
  else:
    missing_words_score = randint(0,3)

  delay_score = score_normalization(delay_score, SCALE)
  speed_score = score_normalization(speed_score, SCALE)
  sge_score = score_normalization(sge_score, SCALE)
  missing_words_score = score_normalization(missing_words_score, SCALE)
  verbatim_score = score_normalization(verbatim_score, SCALE)
  scores = [delay_score, speed_score, sge_score, missing_words_score, verbatim_score]
  
  for i in range(5):
    rating_list[i].append(scores[i])
    

p = np.asarray(rating_list)

for i in p:
  c = np.column_stack((c, i))



np.set_printoptions(precision=4, suppress=True)
print("====== SCORES =====")
print("delay score:", min(c[:,6]), max(c[:,6]), np.mean(c[:,6]), np.std(c[:,6]))
print("speed score:", min(c[:,7]), max(c[:,7]), np.mean(c[:,7]), np.std(c[:,7]))
print("sge score:", min(c[:,8]), max(c[:,8]), np.mean(c[:,8]), np.std(c[:,8]))
print("missing words scores:", min(c[:,9]), max(c[:,9]), np.mean(c[:,9]), np.std(c[:,9]))
print("verbatim score:", min(c[:,10]), max(c[:,10]), np.mean(c[:,10]), np.std(c[:,10]))

print("====== Actual Values =====")
print("delay:", min(c[:,0]), max(c[:,0]))
print("speed:", min(c[:,1]), max(c[:,1]))
print("sge:", min(c[:,2]), max(c[:,2]))
print("missing words:", min(c[:,3]), max(c[:,3]))
print("verbatim:", min(c[:,4]), max(c[:,4]))
print("PF factor:", min(c[:,5]), max(c[:,5]))


print(c.shape) # For a matrix with n rows and m columns, shape will be (n,m)
filename = str(SCALE) + '_gen_dt_' + str(DATASIZE) + '.csv'
with open(filename, 'w') as mf:
  wr = csv.writer(mf)
  for i in c:
    wr.writerow(i)
