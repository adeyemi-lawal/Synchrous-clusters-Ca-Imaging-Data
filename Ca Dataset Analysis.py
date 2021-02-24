#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
from quantities import s


# In[15]:


xls1 = pd.ExcelFile("/Users/Yemi/Downloads/data_part1.xls")
xls2 = pd.ExcelFile("/Users/Yemi/Downloads/data_part2.xls")


# In[16]:


data_sheet1 = xls1.parse(0)
data_sheet2 = xls2.parse(0)


# In[17]:


spike_matrix = []
bad_neurons = [188, 312, 314, 337]


# In[18]:


def computeWeight(value, mean, sd):
    base = .9
    sds_from_mean = (value - mean)/sd
    return base ** sds_from_mean


# In[19]:


#iterate through data and populate spike_matrix with times
#first part of data
neuron_num = 0
for i in range(0, 167):
    sum_square_diff = 0
    sum_nums = 0
    col = data_sheet1.iloc[:,i]
    column = np.array(col)
    train = []
    count = 0
    neuron_num += 1
    for j in range(0, len(column)):
        sum_nums += column[j]
        mean = sum_nums / (count + 1)
        sum_square_diff += ((column[j] - mean) ** 2)
        sd = math.sqrt((sum_square_diff / (count + 1)))
        if(column[j] > (mean + (3 * sd))):
            sum_nums -= column[j]
            sum_nums += column[j] * computeWeight(column[j], mean, sd)
            sum_square_diff -= ((column[j] - mean) ** 2)
            mean = sum_nums / (count + 1)
            sum_square_diff += ((column[j] - mean) ** 2)
            train.append(j * .1)
        count += 1
    spike_matrix.append(train)


# In[20]:


#second part of data
for i in range(0,196):
    sum_square_diff = 0
    sum_nums = 0
    train = []
    neuron_num += 1
    if neuron_num in bad_neurons:
        spike_matrix.append(train)
    col = data_sheet2.iloc[:,i]
    column = np.array(col)
    count = 0
    for j in range(len(column)):
        sum_nums += column[j]
        mean = sum_nums / (count + 1)
        sum_square_diff += ((column[j] - mean) ** 2)
        sd = math.sqrt((sum_square_diff / (count + 1)))
        if(column[j] > (mean + (3 * sd))):
            sum_nums -= column[j]
            sum_nums += column[j] * computeWeight(column[j], mean, sd)
            sum_square_diff -= ((column[j] - mean) ** 2)
            mean = sum_nums / (count + 1)
            sum_square_diff += ((column[j] - mean) ** 2)
            train.append(j * .1)
        count += 1
    spike_matrix.append(train)


# In[21]:


#map every neuron to a color
#convert R to hexadecimal, do the same for G and B
hex_codes = []
path = '/Users/Yemi/Downloads/colors.txt'
with open(path) as colors:
    line = colors.readline()
    while line:
        sent = line.strip()
        hex_num = sent[len(sent) - 6:]
        hex_codes.append(hex_num)
        line = colors.readline()


# In[62]:


bin_mfr = {}
#map bin number to list of tuples of (neuron #, mean firing rates)
for i in range(0, 367):
    times = spike_matrix[i]
    #last[0] = last time in same bin, used for finding diff between firing times
    #last[1] = bin number of last time
    last = [-1,-1]
    sum_diff = 0
    count_nums = 0
    for t in times:
        bin_num = math.floor(t) + 1
        if(bin_num == last[1]):
            sum_diff += (t-last[0])
        else:
            if count_nums >= 2:
                bin_mfr[last[1]] = bin_mfr.get(last[1], []) + [(i, (sum_diff/(count_nums - 1)), count_nums - 1)]
            count_nums = 0
            sum_diff = 0
        
        count_nums += 1
        last[0] = t
        last[1] = bin_num
        


# In[113]:


#in bin_mfr, bin # if mapped to list of tuple(neuron number, mfr, # of intervals used to calc mfr)
#iterate through each bin
for i in range(0, 12):
    bin_mfr[i] = bin_mfr.get(i, [])
    for mfr in bin_mfr[i]: #iterate through each list of tuples for each bin #
        #plot only the mfr from each bin, DONT plot the neuron number, that wouldnt make sense
        plt.plot(i, mfr[1], marker = '.', markersize = mfr[2] * 1.4, color = '#' + hex_codes[mfr[0]])
        
print(bin_mfr[4])

plt.xlabel('Bin Number')
plt.ylabel('Mean Firing Rate')
plt.show()
        


# In[ ]:




