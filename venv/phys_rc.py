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

#definition of the lorenz system
def lorenz(t, xyz, a, b, c):
    x = xyz
    dxdt = np.empty(3)
    dxdt[0] = a * (x[1] - x[0])
    dxdt[1] = x[0] * (b - x[2]) - x[1]
    dxdt[2] = x[0] * x[1] - c * x[2]
    return dxdt


def nonlin(x):
    return 40 * x / (1 + np.abs(x))



#mask size
Nv = 30
#timedomain
tmax = 100

ddt = 0.02
training = int(tmax / ddt)
washout = training // 200
#constant K
K = 0.02

# delay
m = 1 + Nv
m = int(m)

#lorenz system parameters
a, b, c = 10.0, 30.0, 2

#initial value
x0 = 10 * (np.random.uniform(0, 1, size=(3, )) - 1 / 2) #np.array([1.0, 0.0, 0.0])

#timeaxis
t = np.linspace(0, tmax, training)

#data creation
sol = solve_ivp(lorenz, (0, tmax), x0, method='RK45', t_eval=t, args=(a, b, c))
time = np.array(sol.t)
data0 = np.array(sol.y)[:, washout:]
x, y, z = data0

x = (x - x.mean()) / x.std()
y = (y - y.mean()) / y.std()
z = (z - z.mean()) / z.std()

xnew = 10 * (np.random.uniform(0, 1, size=(3, )) - 1 / 2)
sol = solve_ivp(lorenz, (0, tmax), xnew, method='RK45', t_eval=t, args=(a, b, c))
data1 = np.array(sol.y)[:, washout:]
c1, c2, c3 = data1

c1 = (c1 - x.mean()) / x.std()
c2 = (c2 - y.mean()) / y.std()
c3 = (c3 - z.mean()) / z.std()

mask = np.random.uniform(0, 1, size=(Nv, ))
G = 0.03

def state_mat(delay, Nv, data, mask, const, fun=nonlin):
    steps = data.shape[0]
    J = np.empty(Nv * steps + Nv)
    for i in range(steps):
        for j in range(Nv):
            J[Nv * i + j] = G * data[i] * mask[j]  # + G * x[k - d] * M[K]  # + 0.85

    x_out = np.zeros(Nv * steps + Nv)
    x_out[:Nv] = np.random.uniform(0, 1, size=Nv)

    for k in range(Nv, Nv * steps + Nv):
        x_out[k] = fun(const * x_out[k - delay] + J[k])

    S = np.empty((steps, Nv))
    for i in range(steps):
        for j in range(Nv):
            S[i, j] = x_out[Nv * i + j]
    return S


S = state_mat(m, Nv, x[:-1], mask, K, nonlin)

# computation of the output matrix via Tikhonov regularization with regularization term beta
beta = 1e-10
target = x[1:]
Wout_x = np.linalg.lstsq(S.T @ S + beta * np.identity(Nv), S.T @ target, rcond=None)[0]

print('norm = ', np.linalg.norm(S @ Wout_x - target))
S = state_mat(m, Nv, c1[:-1], mask, K, nonlin)
x_pred = S @ Wout_x

def nrmse(x,y):
    return np.linalg.norm(x-y)/(np.sqrt(len(y))*y.std())

print(c1[:-1].shape[0], x[:-1].shape[0])

print('NRMSE x-component: %5.5f\n' % nrmse(c1[1:], x_pred))


plt.plot(time[washout + 1 :], c1[1:], label='sol')
plt.plot(time[washout + 1 :], x_pred * x.std() + x.mean(), label='pred')

plt.legend()
plt.show()




