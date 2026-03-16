import random
from transformers import GPT2TokenizerFast
import math 
import Stemmer
import numpy as np
import os
import pickle



feedforward_matrices = []
d_model = 512
d_size = 50257
fan_in = d_model
fan_out = d_model
qkv = 3*8*64
n = 6

def matricies_init_ ():
    dictonary_vectors = np.array([[ random.uniform(-1, 1) for _ in range(d_model)] for _ in range(d_size)])
    multihead_matrices = np.array([[[ random.uniform(- math.sqrt(6/(fan_in+fan_out)), math.sqrt(6/(fan_in+fan_out))) for _ in range(qkv)] for _ in range(d_model)] for _ in range(n)])
    return dictonary_vectors, multihead_matrices
dictonary_vectors, multihead_matrices = matricies_init_ ()
print(multihead_matrices[0][0])
