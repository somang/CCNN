import csv
import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
from random import randint
import math
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=0, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

DATASIZE = 100000
### normal distributed score
# Binomial(n, p) ~ Normal(n*p, sqrt(n*p*(1-p)))
trn = get_truncated_normal(mean=4.66, sd=2.53) # v1(4.919,2.19) v2(4.4,2.87)
delay_score = np.rint(trn.rvs(DATASIZE))

trn = get_truncated_normal(mean=4.33, sd=2.57) # v1(4.66,2.39), v2(4,2.75)
wpm_score = np.rint(trn.rvs(DATASIZE))

trn = get_truncated_normal(mean=4.26, sd=2.32) # v1(4.33,2.15) v2(4.18,2.49)
mw_score = np.rint(trn.rvs(DATASIZE))

trn = get_truncated_normal(mean=4.53, sd=2.19) # v1(4.62,2.10) v2(4.44,2.28)
sge_score = np.rint(trn.rvs(DATASIZE))

trn = get_truncated_normal(mean=4.20, sd=2.51) # v1(4.17,2.20) v2(4.22,2.81)
verbatim_score = np.rint(trn.rvs(DATASIZE))

