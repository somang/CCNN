from scipy.stats import truncnorm


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import scipy
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.mlab import normpdf

"""
def get_truncated_normal(mean=0, sd=0, low=0, high=10):
  value = truncnorm((low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd)
  return value

# delay, wpm, similarity, number of errors
### normal distribution using the mean and sd from existing data.
trn = get_truncated_normal(mean=4895.75, sd=1477.94, low=0, high=12000) # 11910 was the max
r_delay = trn.rvs(100000)
"""
x = np.linspace(0, 12000, 100)

mu = 4895.75
sd = 1477.94
pdf = mlab.normpdf(x, mu, sd)
plt.axvline(x=mu, color='b') # mean
plt.axvline(x=mu - 2*sd, color='g') # -1sd
plt.axvline(x=mu + 2*sd, color='g') # 1sd


plt.plot(x, pdf, color='r') 
title = "mu = {},  SD = {}".format(mu, sd)
plt.ylabel('probability')
plt.title(title)
plt.show()

