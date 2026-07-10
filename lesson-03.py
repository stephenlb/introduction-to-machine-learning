# Introduction to Machine Learning
import numpy as np

print("Slicing and Indexing")
print()


a = np.arange(10)
print(a)
print(a[ 5])  ## Fith Element
print(a[-1])  ## Last Element
print(a[-2])  ## Second to last Element
print(a[-2])  ## Second to last Element
print(a[0:5]) ## Slice first five elemetns
print(a[ :5]) ## Slice first five elemetns
print(a[::2]) ## Slice Every other Element
print(a[::5]) ## Slice Every 5th Element

print(a > 5) ## Bool conditional
print(a[a>5])

## 2x10 Matrix
m = np.array([
    np.arange(10),
    np.arange(10) + 10,
])
print(m)
print(m[1][1]) #11
print(m[1][0]) #10
print(m[1, 0]) #10
print(m[:, 0]) # 0 10
print(m[:, 1]) # 1 11
print(m[:, 2]) # 2 12




