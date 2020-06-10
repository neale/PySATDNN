#############################
# utils.py
# various core utils that allow us to:
# generate data,
# time functions
# interact with pytorch
#############################

import numpy as np
import torch
from functools import wraps
from time import time


def timed(f):
  @wraps(f)
  def wrapper(*args, **kwds):
    start = time()
    result = f(*args, **kwds)
    elapsed = time() - start
    print ("%s took %d time to finish" % (f.__name__, elapsed))
    return result
  return wrapper


def to_torch_tensor(x, y, w, b):
    x = [[float(char) for char in item] for item in x]
    x = torch.tensor(x)
    weights = []
    biases = []
    for weight_set, bias_set in zip(w, b):
        weights.append(torch.tensor(weight_set))
        biases.append(torch.tensor(bias_set))
    w = weights
    b = biases
    y = torch.tensor(y).float()
    return (x, y, w, b)


def generate_iid_samples(dim, n_samples, unsat=False):
    upper_bound = 2**dim
    samples = np.random.randint(0, upper_bound, n_samples)
    samples = [sample.item() for sample in samples]
    # labels assigned to half of the cube
    if unsat:
        y = [1 if np.random.uniform() < .5 else 0 for sample in samples]
    else:
        y = [1 if sample < (upper_bound/2) else 0 for sample in samples]
    return samples, y 

