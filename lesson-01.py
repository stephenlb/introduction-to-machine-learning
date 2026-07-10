## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Introduction to Machine Learning
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import numpy as np

## 1D Array = Vector
vector = np.array([1.0, 2.0, 3.0])
#print(vector)
#print(vector.shape)
#print(vector.dtype)

## 2D Array = Matrix
matrix = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])
#print(matrix)
#print(matrix.shape)
#print(matrix.dtype)

## 3D Array = Cube """3D Vector"""
cube = np.array([
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ],
    [
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ],
])
#print(cube)
#print(cube.shape)
#print(cube.dtype)


## Generate Tensors 
zeros = np.zeros((5,))
ones = np.ones((10,10))
#print(zeros)
#print(ones)

## Array Range
arange = np.arange(0,10,1)
print("arange")
print(arange)
print()

## Linspace
linspace = np.linspace(0,10,10)
print("linspace")
print(linspace)
print()





