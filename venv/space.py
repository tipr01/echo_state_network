import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



#activation function
def act(x):
    return np.tanh(x)

#function that creates a random adjacency matrix W_r with spectral radius less than lmd
def random_adjacency_matrix(n, p, lmd):
    # matrix = np.random.choice([-1, 0, 1], size=(n, n), p=[0, 0.9, 0.1])
    matrix = np.random.uniform(-1, 1, size=(n, n))

    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0

    for i in range(n):
        for j in range(n):
            if np.random.random() < 1 - p:
                matrix[j][i] = 0
            matrix[i][j] = matrix[j][i]

    # If i is connected to j, j is connected to i
    # for i in range(n):
    #     for j in range(n):
    #         if np.abs(matrix[j][i]) == 1 or np.abs(matrix[i][j]) == 1:
    #             if (i+j % 2) == 0:
    #                 matrix[i][j], matrix[j, i] = 1, 1
    #             else:
    #                 matrix[i][j], matrix[j, i] = -1, -1
    #         else:
    #             matrix[i][j] = matrix[j][i]

    # matrix = np.reshape(matrix, (n,n))


    # set spectral radius to a constant near 1, less than 1 ensures the echo state property
    matrix = (lmd / np.max(np.real(np.linalg.eigvals(matrix)))) * matrix
    return matrix


#definition of the lorenz system
def lorenz(t, xyz, a, b, c):
    x = xyz
    dxdt = np.empty(3)
    dxdt[0] = a * (x[1] - x[0])
    dxdt[1] = x[0] * (b - x[2]) - x[1]
    dxdt[2] = x[0] * x[1] - c * x[2]
    return dxdt

