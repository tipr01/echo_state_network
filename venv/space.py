import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random




#lorenz system parameters
a, b, c = 10.0, 30.0, 2


#critical points
crit_pnt1 = np.array([np.sqrt(c * (b - 1)), np.sqrt(c * (b - 1)), b - 1])
crit_pnt2 = np.array([ - np.sqrt(c * (b - 1)), - np.sqrt(c * (b - 1)), b - 1])


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


def nonlin(x):
    return x / (1 + np.abs(x))

def test_func(x, b, normal, c):
    return 2 / np.pi * np.arctan(c * np.dot(normal, x - b))


def washout(reservoir, washout, n, m, data, map, K, j):
    for i in range(n, n * washout):
        if i // n > j:
            j += 1
        k = i % n
        reservoir = np.append(reservoir, nonlin(K * reservoir[i - m] + data[:, j].T @ map[k, :]))
    return reservoir, j

def training(reservoir, washout, steps, n, m, data, map, K, j):
    for i in range(n * washout, n * steps):
        if i // n > j:
            j += 1
        k = i % n
        reservoir = np.append(reservoir, nonlin(K * reservoir[i - m] + data[:, j].T @ map[k, :]))
    reservoir_state_matrix = np.reshape(reservoir[n * washout: n * steps], (steps - washout, n)).T
    return reservoir_state_matrix


def prediction(R, n, m, steps, prediction, j, res, Wout_x, K, map):
    reservoir = res
    d = Wout_x @ reservoir
    for i in range(n, n * prediction):
        if i // n > j:
            j += 1
            d = Wout_x @ np.array(reservoir[-n: ])
            R.append(reservoir[-n: ])
        k = i % n
        print(np.shape(reservoir))
        reservoir = np.append(reservoir, nonlin(K * reservoir[i - m] + d @ map[k, :]))
    R = np.array(R).T
    x = Wout_x @ R
    return x








