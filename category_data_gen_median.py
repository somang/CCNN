
from scipy.stats import norm
import csv
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from scipy.stats import truncnorm
import math

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




"""
CTV news: delay, wpm, sim_value, spelling, mw, wd,
4895.75 1477.93 [7271, 2669] [4075, 4669.5, 5775]
232.03 200.48 [814.159, 57.143] [118.56, 143.21, 313.46]
0.854 0.1045 [1, 0.689] [0.7770, 0.8416, 0.9467]
0.0 0.0 [ 0.] [ 0.]
4.833 6.243 [20, 0] [0.75, 1.5, 7]
12.333 11.2496 [39, 0] [1.75, 11, 20]

Citytv news: delay, wpm, sim_value, spelling, mw, wd,
7992.79 1888.86 [11910, 3319] [6745, 7690, 9070.5]
132.20 77.16 [355.17, 48.37] [67.891, 131.165, 161.672]
0.817 0.290 [1, 0] [0.8292, 0.9366, 0.9724]
5.21052631579 7.35261462531 [ 24.] [ 0.] [0.5, 2, 6.5]
8.84210526316 11.0132129157 [ 34.] [ 0.] [0.5, 2, 14]
"""

# Generate an array of 200 random sample from a normal dist with 
# mean 0 and stdv 1
sample = np.asarray([
  # video 1
  [3319],[6820],[7690],[6470],[7380],
  [11030],[6670],[9280],[6441],[6941],
  [8941],[9620],[7541],[8820],[8810],
  [11910],[8760],[6220],[9200],[7992.78],
  [1888.8578],[6745],[9070.5],
  # video2
  [3090],[4859],[6930],[4460],[2669],
  [7271],[4480],[4920],[4340],[7060],
  [5390],[3280],[4895.75],[1477.93],
  [2669],[4075],[4669.5],[5775]
 ])

print(np.mean(sample), np.std(sample), np.percentile(sample,[25,50,75])) # [ 4480.    6670.    7992.78]

# Distribution fitting
# norm.fit(data) returns a list of two parameters 
# (mean, parameters[0] and std, parameters[1]) via a MLE approach 
# to data, which should be in array form.
parameters = norm.fit(sample)

# now, parameters[0] and parameters[1] are the mean and 
# the standard deviation of the fitted distribution
x = np.linspace(-5, 15000, 100)

# Generate the pdf (fitted distribution)
fitted_pdf = norm.pdf(x,loc = parameters[0],scale = parameters[1])

# Type help(plot) for a ton of information on pyplot
plt.plot(x,fitted_pdf,"red",label="Fitted normal dist",linestyle="dashed", linewidth=1)
plt.hist(sample,normed=1,color="cyan",alpha=.3) #alpha, from 0 (transparent) to 1 (opaque)
# insert a legend in the plot (using label)
plt.legend()

# we finally show our work
plt.show()







