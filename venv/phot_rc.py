import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random
import scipy.io

# lorenz system parameters
a, b, c = 10.0, 30.0, 2

# critical points
crit_pnt1 = np.array([np.sqrt(c * (b - 1)), np.sqrt(c * (b - 1)), b - 1])
crit_pnt2 = np.array([- np.sqrt(c * (b - 1)), - np.sqrt(c * (b - 1)), b - 1])


# activation function
def act(x):
    return np.arctan(20 * x)

# definition of the lorenz system
def lorenz(t, xyz, a, b, c):
    x = xyz
    dxdt = np.empty(3)
    dxdt[0] = a * (x[1] - x[0])
    dxdt[1] = x[0] * (b - x[2]) - x[1]
    dxdt[2] = x[0] * x[1] - c * x[2]
    return dxdt



def nonlin(x):
    return 40 * x / (1 + x)

# def swing_fun(swing, m, K, delay_divisor, delay_remainder, G, nonlin_fun=nonlin):
#     initial_state_matrix = np.zeros((swing, m))
#     initial_state = np.random.uniform(0, 1, size=m)
#     initial_state_matrix[0, :] = initial_state
#
#     for j in range(1, swing):
#         for i in range(m):
#             initial_state_matrix[j, i] = nonlin_fun(K * initial_state_matrix[j - delay_divisor, i - delay_remainder] + G)
#
#     return initial_state_matrix[-1, :]

def state_mat(U, mask, delay, G, K, nonlin_fun=nonlin):
    m = mask.shape[0]
    length = U.shape[0]
    S = np.ones((length, m + 1))
    delay_divisor = delay // m
    delay_remainder = delay % m

    for j in range(1, length):
        for i in range(m):
            S[j, i] = nonlin_fun(K * S[j - delay_divisor, i - delay_remainder] + G * mask[i] * U[j])
    return S

def state_mat_multi_data(data, mask, delay, G, K=0.02, nonlin_fun=nonlin):
    m = mask.shape[0]
    length = data.shape[1]
    swing = 100
    S = np.zeros((length, m))
    delay_divisor = delay // m
    delay_remainder = delay % m

    initial_state_matrix = np.zeros((swing, m))
    initial_state = np.random.uniform(0, 1, size=m)
    initial_state_matrix[0, :] = initial_state

    for j in range(swing):
        for i in range(m):
            if j * m + i - delay >= 0:
                initial_state_matrix[j, i] = nonlin_fun(K * initial_state_matrix[j - delay_divisor, i - delay_remainder] + G * np.dot(mask[i], np.ones(3)))

    S[0, :] = initial_state_matrix[-1, :]

    for j in range(1, length):
        for i in range(m):
            S[j, i] = nonlin_fun(K * S[j - delay_divisor, i - delay_remainder] + G * np.dot(mask[i], data[:, j]))
    #S = np.concatenate((S, np.ones(length).reshape((length, 1))), axis=1)
    return S

def state_mat_autoregr(initialization, init_vec, W, mask, delay, anz, G, nonlin_fun=nonlin, K=0.02):
    m = mask.shape[0]
    length = anz
    S = np.ones((length, m + 1))
    delay_divisor = delay // m
    delay_remainder = delay % m

    #S[0, :] = swing_fun(swing, m, K, delay_divisor, delay_remainder, G)
    S[0, :] = init_vec
    d = initialization
    for j in range(1, length):
        for i in range(m):
            S[j, i] = nonlin_fun(K * S[j - delay_divisor, i - delay_remainder] + G * mask[i] * d)
        d = S[j, :] @ W
    return S


def nrmse(x, y):
    return np.linalg.norm(x - y) / (np.sqrt(len(y)) * y.std())

def tikhonov(S, target, beta=5e-6):
    n = np.shape(S)[1]
    return np.linalg.lstsq(S.T @ S + beta * np.identity(n), S.T @ target, rcond=None)[0]

datasize = 35000

# discretization
dt = 0.02

#timedomain
tmax = int(datasize * dt)

#initial value
x0 = np.array([1.0, 0.0, 0.0])

#time series stepsize
stepsize = 1e-3
steps = int(tmax / stepsize)

#timeaxis
t = np.linspace(0, tmax, steps)

#data creation
sol = solve_ivp(lorenz, (0, tmax), x0, method='RK45', t_eval=t, args=(a, b, c))
time = np.array(sol.t)
data = np.array(sol.y)

indices = [int(steps / datasize * i) for i in range(datasize)]
x, y, z = data
xnew = np.take(x, indices)
ynew = np.take(y, indices)
znew = np.take(z, indices)

data_new = xnew, ynew, znew

file = 'coordinate_data.mat'
scipy.io.savemat(file, mdict={'out': data_new}, oned_as='row')

# python solved prediction
sol = solve_ivp(lorenz, (0, tmax), data[:, -1], method='RK45', t_eval=t, args=(a, b, c))
time_pred = np.array(sol.t)
pred = np.array(sol.y)

indices = [int(steps / datasize * i) for i in range(datasize)]
x, y, z = pred
xnew = np.take(x, indices)
ynew = np.take(y, indices)
znew = np.take(z, indices)

pred_new = xnew, ynew, znew

file = 'coordinate_pred.mat'
scipy.io.savemat(file, mdict={'out': pred_new}, oned_as='row')