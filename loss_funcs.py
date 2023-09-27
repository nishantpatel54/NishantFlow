import numpy as np

def mse(y,propose_y):
    return np.mean(np.power(y-propose_y, 2))

def mse_prime(y,propose_y):
    return 2*(y-propose_y) / np.size(y)

def cross_entropy(y,propose_y):
    return np.mean(-y * np.log(propose_y) - (1-y) * np.log(1-propose_y))

def cross_entropy_prime(y,propose_y):
    return ((1-y)/(1-propose_y) - y/propose_y) / np.size(y)