import csv
import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
from random import randint
import math
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=0, low=0, high=10):
  return truncnorm(
    (low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd)

DATASIZE = 10
### normal distributed score
# Binomial(n, p) ~ Normal(n*p, sqrt(n*p*(1-p)))
'''
trn = get_truncated_normal(mean=4.66, sd=2.53) # v1(4.919,2.19) v2(4.4,2.87)
delay_score = np.rint(trn.rvs(DATASIZE))
delay_score = np.sort(delay_score)

trn = get_truncated_normal(mean=4.33, sd=2.57) # v1(4.66,2.39), v2(4,2.75)
wpm_score = np.rint(trn.rvs(DATASIZE))
wpm_score = np.sort(wpm_score)

trn = get_truncated_normal(mean=4.53, sd=2.19) # v1(4.62,2.10) v2(4.44,2.28)
sge_score = np.rint(trn.rvs(DATASIZE))
sge_score = np.sort(sge_score)

trn = get_truncated_normal(mean=4.26, sd=2.32) # v1(4.33,2.15) v2(4.18,2.49)
mw_score = np.rint(trn.rvs(DATASIZE))
mw_score = np.sort(mw_score)

trn = get_truncated_normal(mean=4.20, sd=2.51) # v1(4.17,2.20) v2(4.22,2.81)
verbatim_score = np.rint(trn.rvs(DATASIZE))
verbatim_score = np.sort(verbatim_score)
'''

## actual values:
trn = get_truncated_normal(mean=4895.75, sd=1477.94, low=0, high=12000)
delay_value = trn.rvs(DATASIZE)
delay_value = np.flip(np.sort(delay_value),0)

trn = get_truncated_normal(mean=232.03, sd=200.48, low=0, high=850)
speed_value = trn.rvs(DATASIZE)
speed_value = np.sort(speed_value)

trn = get_truncated_normal(mean=0.85, sd=0.2, low=0.0, high=1.0)
sent_similarity = trn.rvs(DATASIZE)
sent_similarity = np.sort(sent_similarity)

trn = get_truncated_normal(mean=1, sd=2, low=0.0, high=10)
sge = np.rint(trn.rvs(DATASIZE))
sge = np.sort(sge)

trn = get_truncated_normal(mean=5.02, sd=6.79, low=0.0, high=25)
mw = np.rint(trn.rvs(DATASIZE))
mw = np.sort(mw)

print(delay_value)

# paraphrasing factor:
pf_factors = []
pf = 0
for i in range(len(sent_similarity)):
  if i < 1.0: #if sentence similarity is not 100, then
    missing_word_count = mw[i]
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

# stack all of them
c = np.column_stack((delay_value, speed_value))
c = np.column_stack((c, sent_similarity))
c = np.column_stack((c, sge))
c = np.column_stack((c, mw))
c = np.column_stack((c, pf_factors))
'''
c = np.column_stack((c, delay_score))
c = np.column_stack((c, wpm_score))
c = np.column_stack((c, sge_score))
c = np.column_stack((c, mw_score))
c = np.column_stack((c, verbatim_score))
'''
#np.random.shuffle(c) # shuffle the order in rows

for i in c:
  print(i)
np.set_printoptions(precision=4, suppress=True)
print(c.shape)
for i in c:
  print(i)