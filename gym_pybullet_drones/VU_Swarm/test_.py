import numpy as np


x = np.array([1,1])

y = np.array([2,3,4,5,6])

print(np.concatenate((y[:2], x)))