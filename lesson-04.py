import numpy as np

## Aggregations

a = np.array([[1., 2., 3.],
              [4., 5., 6.]])
print(a)

print(a.sum())
print(a.max())
print(a.mean())
print(a.sum(axis=1))
print(a.argmax()) ## index of the largest value
