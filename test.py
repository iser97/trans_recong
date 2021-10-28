import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

d = open('./cache/train_data.txt', mode='r')
data = np.array(json.loads(d.readlines()[0]))
for index, d in enumerate(data):
    d = d.reshape((8, 8))
    plt.imsave('{}.jpg'.format(index), d)
