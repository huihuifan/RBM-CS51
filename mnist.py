%matplotlib inline

import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_mldata

#Obtain complete MNIST images
DATA_PATH = '~/data'
mnist = fetch_mldata('MNIST original', data_home=DATA_PATH)

print mnist.data.shape

subset = mnist.data[:100]

#Convert to binary matrix 
for img in subset:
    img[img < 100] = 0
    img[img > 0] = 1

"""
Each image is stored as a one dimensional array, so you will
need to reshape with the following code:

image.reshape(28, 28)

for each image within the subset
"""