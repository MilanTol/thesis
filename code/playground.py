import numpy as np
from halo_model.algorithms.ridders_derivative import ridders_derivative


def f(x):
    return np.sin(x)

x = 6
h_init = 1

h = h_init
pairs = []  
order = 2
for i in range(order):
    pair = [f(x-h), f(x+h)]
    pairs.append(pair)
    h /= 2
pairs = np.array(pairs)

deriv = ridders_derivative(pairs, h_init, 2, 1e-5)
print(deriv)