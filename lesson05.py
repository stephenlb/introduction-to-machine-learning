import numpy as np

def dot_product():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.5, -1.0, 2.0])
    print(a, b)
    print(a.dot(b))
    print(a @ b)


def dot_product2():
    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    b = np.array([
        [0.5, -1.0],
        [0.5, -1.0],
        [0.5, -1.0],
    ])
    print(a)
    print(b)
    print(a.dot(b))
    print(a @ b)

def matrix_mult():
    vector = np.array([2.0, 2.0, 3.0])
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    print(vector.shape)
    print(matrix.shape)
    print(vector @ matrix)

def transpose():
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    print(matrix.shape)
    print(matrix)

    print(matrix.T.shape)
    print(matrix.T)


def identity():
    I = np.eye(3)
    print(I)
    A = np.array([
        [3.0, 2.0],
        [1.0, 2.0],
    ])
    print(A)
    b = np.array([5.0, 5.0])
    print(b)

    x = np.linalg.solve(A, b)
    print(x)



