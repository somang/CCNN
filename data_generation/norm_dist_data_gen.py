import csv
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from random import gauss
from scipy.stats import truncnorm
import math

SCALE = 10
DATASIZE = 100000
print("SCALE:",SCALE,", SIZE:",DATASIZE)

def setCategoryScore(index, p, score):
  if index == score:
    RATING_LIST[p].append(1)
  else:
    RATING_LIST[p].append(0)

#https://crtc.gc.ca/eng/archive/2012/2012-362.htm
def get_truncated_normal(mean=0, sd=0, low=0, high=10):
  value = truncnorm((low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd)
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

# delay, wpm, similarity, number of errors
### normal distribution using the mean and sd from existing data.
trn = get_truncated_normal(mean=4895.75, sd=1477.94, low=0, high=12000)
r_delay = trn.rvs(DATASIZE)

trn = get_truncated_normal(mean=232.03, sd=200.48, low=0, high=850)
r_wpm = trn.rvs(DATASIZE)

trn = get_truncated_normal(mean=1, sd=2, low=0.0, high=10)
r_spell_grammar_errors = np.rint(trn.rvs(DATASIZE))

trn = get_truncated_normal(mean=0.85, sd=0.2, low=0.0, high=1.1)
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

plt.hist(r_missing_words, bins='auto')  
#plt.show()


###### Simulated scores based on the fact generated from previous. ######
rating_list_10 = [
  [],[],[],[],[],
  [],[],[],[],[],[],[],[],[],[],[],
  [],[],[],[],[],[],[],[],[],[],[],
  [],[],[],[],[],[],[],[],[],[],[],
  [],[],[],[],[],[],[],[],[],[],[],
  [],[],[],[],[],[],[],[],[],[],[]
]
rating_list_5 = [
  [],[],[],[],[],
  [],[],[],[],[],
  [],[],[],[],[],
  [],[],[],[],[],
  [],[],[],[],[],
  [],[],[],[],[],
]
rating_list_3 = [
  [],[],[],[],[],
  [],[],[],
  [],[],[],
  [],[],[],
  [],[],[],
  [],[],[],
]
bins = {10:rating_list_10, 5:rating_list_5, 3:rating_list_3}
RATING_LIST = bins[SCALE]
#print(len(RATING_LIST))
mw_trn = get_truncated_normal(mean=4.26, sd=2.32, low=0.0, high=10)
# [delay], [speed], [verbatim factor score], [spelling and grammar error score], [missing words score] 
for i in c:
  delay_score, speed_score, verbatim_score, sge_score, missing_words_score = 0,0,0,0,0
  # calculate delay rating
  delay = i[0]
  wpm = i[1]
  spell_grammar_errors = i[2]
  missing_words = i[3]
  sentence_sim = i[4]
  
  #  delay_score: mean=4.66, sd=2.53
  if delay <= 2600:
    delay_score = randint(9, 10)
  elif 2600 < delay <= 4000:
    delay_score = randint(6, 8)
  elif 4000 < delay <= 5000:
    delay_score = randint(4, 6)
  else:
    delay_score = randint(0, 4)

  # wpm_score: mean=4.33, sd=2.57
  # calculate speed_rating
  if wpm <= 45:
    speed_score = randint(0,3)
  elif 45 < wpm <= 90:
    speed_score = randint(0,3) # when its too slow to read
  elif 90 < wpm <= 140:
    speed_score = randint(5,10)
  elif 140 < wpm <= 220:
    speed_score = randint(7,10)
  elif 220 < wpm <= 400:
    speed_score = randint(4,7)
  else:
    speed_score = randint(0,3)

  # sge_score = mean=4.53, sd=2.19
  if spell_grammar_errors == 0:
    sge_score = 10
  elif 0 < spell_grammar_errors <= 3:
    sge_score = round(gauss(4.55, 1.1))
  else:
    sge_score = randint(0,1)
  
  # mw_score: mean=4.26, sd=2.32
  if missing_words == 0:
    missing_words_score = 10
  elif 0 < missing_words <= 13:
    missing_words_score = round(gauss(4.26,0.3))
  else:
    missing_words_score = 0
  
  # verbatim_score: mean=4.20, sd=2.51
  # Paraphrasing (verbatimness) score which audiences subjectively feel
  if sentence_sim == 1.0:
    verbatim_score = 10
  elif 0.7 < sentence_sim < 1.0:
    if missing_words == 0:
      verbatim_score = 10
    elif 0 < missing_words <= 15:
      verbatim_score = round(gauss(4.50,0.3))  
  else:
    verbatim_score = round(gauss(1.0,0.6))

  delay_score = score_normalization(delay_score, SCALE)
  speed_score = score_normalization(speed_score, SCALE)
  sge_score = score_normalization(sge_score, SCALE)
  missing_words_score = score_normalization(missing_words_score, SCALE)
  verbatim_score = score_normalization(verbatim_score, SCALE)

  
  chain = 11 if SCALE == 10 else SCALE
  scores = [delay_score, speed_score, sge_score, 
            missing_words_score, verbatim_score]
  for i in range(5): # 0-4, for scores
    score = scores[i]
    RATING_LIST[i].append(score)

  if SCALE == 10:
    for p in range(5,16):
      setCategoryScore(p-5, p, delay_score)
    for p in range(16,27):
      setCategoryScore(p-16, p, speed_score)
    for p in range(27,38): 
      setCategoryScore(p-27, p, sge_score)
    for p in range(38,49): 
      setCategoryScore(p-38, p, missing_words_score)
    for p in range(49,60):
      setCategoryScore(p-49, p, verbatim_score)

  elif SCALE == 5:
    for p in range(5,10): 
      setCategoryScore(p-4, p, delay_score)
    for p in range(10,15):
      setCategoryScore(p-9, p, speed_score)
    for p in range(15,20): 
      setCategoryScore(p-14, p, sge_score)
    for p in range(20,25): 
      setCategoryScore(p-19, p, missing_words_score)
    for p in range(25,30):
      setCategoryScore(p-24, p, verbatim_score)

  elif SCALE == 3:
    for p in range(5,8): 
      setCategoryScore(p-4, p, delay_score)
    for p in range(8,11):
      setCategoryScore(p-7, p, speed_score)
    for p in range(11,14): 
      setCategoryScore(p-10, p, sge_score)
    for p in range(14,17): 
      setCategoryScore(p-13, p, missing_words_score)
    for p in range(17,20):
      setCategoryScore(p-16, p, verbatim_score)

p = np.asarray(RATING_LIST)
#print(p.shape)

for i in p:
  c = np.column_stack((c, i))

np.set_printoptions(precision=4, suppress=True)
'''
for i in c:
  if SCALE == 10:
    print("delay score:", i[6], i[11:22])
    print("speed score:", i[7], i[22:33])
    print("sge score:", i[8], i[33:44])
    print("missing words scores:", i[9], i[44:55])
    print("verbatim score:", i[10], i[55:66])
  elif SCALE == 5:
    print("delay score:", i[6], i[11:16])
    print("speed score:", i[7], i[16:21])
    print("sge score:", i[8], i[21:26])
    print("missing words scores:", i[9], i[26:31])
    print("verbatim score:", i[10], i[31:36])
  elif SCALE == 3:
    print("delay score:", i[6], i[11:14])
    print("speed score:", i[7], i[14:17])
    print("sge score:", i[8], i[17:20])
    print("missing words scores:", i[9], i[20:23])
    print("verbatim score:", i[10], i[23:26])
  print()

'''


"""
  0 delay                                     6 delay scores
  1 speed                                     7 speed scores
  2 spelling and grammar errors               8 spelling and grammar errors scores
  3 missing words                             9 missing words score
  4 sentence similarity: verbatim scores     10 verbatim score
  5 PF values

  delay score in categorical
  11 12 13 14 15 | 16 17 18 19 20 | 21
   0  1  2  3  4 |  5  6  7  8  9 | 10
  
  speed score in categorical
  22 23 24 25 26 | 27 28 29 30 31 | 32
   0  1  2  3  4 |  5  6  7  8  9 | 10
  
  spelling and grammar score in categorical
  33 34 35 36 37 | 38 39 40 41 42 | 43
   0  1  2  3  4 |  5  6  7  8  9 | 10

  missing words score in categorical
  44 45 46 47 48 | 49 50 51 52 53 | 54
   0  1  2  3  4 |  5  6  7  8  9 | 10

  verbatim score in categorical
  55 56 57 58 59 | 60 61 62 63 64 | 65
   0  1  2  3  4 |  5  6  7  8  9 | 10
"""


print("====== SCORES =====")
#print("delay score:", np.mean(c[:,6]), np.std(c[:,6]))
#print("speed score:", np.mean(c[:,7]), np.std(c[:,7]))
#print("sge score:", np.mean(c[:,8]), np.std(c[:,8]))
#print("missing words scores:", np.mean(c[:,9]), np.std(c[:,9]))
print("verbatim score:", np.mean(c[:,10]), np.std(c[:,10]))

'''
print("====== Actual Values =====")
print("delay:", #min(c[:,0]), max(c[:,0]), 
      "[4075 4669.5 5775]",
      #np.mean(c[:,0]), np.std(c[:,0]),
      np.percentile(c[:,0], [25,50,75]))

print("speed:", #min(c[:,1]), max(c[:,1]),
      "[118.56 143.21 313.46]",
      #np.mean(c[:,1]), np.std(c[:,1]),
      np.percentile(c[:,1], [25,50,75]))

print("sge:", #min(c[:,2]), max(c[:,2]),
      #np.mean(c[:,2]), np.std(c[:,2]),
      np.percentile(c[:,2], [25,50,75]))
 
print("missing words:", #min(c[:,3]), max(c[:,3]),
      "[0.75  1.5  7]",
      #np.mean(c[:,3]), np.std(c[:,3]),
      np.percentile(c[:,3], [25,50,75]))

print("verbatim:", #min(c[:,4]), max(c[:,4]), 
      "[0.7770  0.8416  0.9467]",
      #np.mean(c[:,4]), np.std(c[:,4]),
      np.percentile(c[:,4], [25,50,75]))
#print("PF factor:", min(c[:,5]), max(c[:,5]),
#      np.mean(c[:,5]), np.std(c[:,5]),
#      np.percentile(c[:,5], [25,50,75]))
'''


print(c.shape) # For a matrix with n rows and m columns, shape will be (n,m)
filename = str(SCALE) + '_nd_dt_' + str(DATASIZE) + '.csv'
with open(filename, 'w') as mf:
  wr = csv.writer(mf)
  for i in c:
    wr.writerow(i)
