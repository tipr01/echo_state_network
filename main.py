import numpy as np
from scipy.integrate import solve_ivp
import random
import matplotlib.pyplot as plt
import mpl_toolkits
from reservoirpy.nodes import Reservoir, Ridge, Input


def random_adjacency_matrix(n):
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]

    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0

    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]

    matrix = np.reshape(matrix, (n,n))
    return matrix


#define lorenz system
def lorenz(t, xyz, a, b, c):
    x = xyz
    dxdt = np.empty(3)
    dxdt[0] = a * (x[1] - x[0])
    dxdt[1] = x[0] * (b - x[2]) - x[1]
    dxdt[2] = x[0] * x[1] - c * x[2]
    return dxdt


#lorenz system parameters
a, b, c = 10.0, 30.0, 2


# sol = np.empty((steps+1, 3))

#initial value
x0 = np.array([0.0, 1.0, 1.0])

#timedomain
tmax = 20

washout = 1000
training = 10000
prediction = 1500

steps = washout + training
t = np.linspace(0, tmax, steps)

sol = solve_ivp(lorenz, (0, tmax), x0, method='RK45', t_eval=t, args=(a, b, c))



# print(time, '\n', x, y, z)



time = np.array(sol.t)
coo = np.array(sol.y)
x, y, z = coo[0], coo[1], coo[2]

# plt.plot(t, x, color='r', label='x')
# plt.plot(t, y, color='g', label='y')
# plt.plot(t, z, color='b', label='z')
#
# plt.show()

plt.figure('lorenz solution')

ax = plt.axes(projection='3d')

# Data for a three-dimensional line
xline = x
yline = y
zline = z
ax.plot3D(xline, yline, zline, 'blue')

# plt.show()



n = 500


r = np.random.rand(n, 1) * 100
r = np.array(r)

# Win = np.random.random((n, 3)) * 100
Win = [[random.choice([0, -c, c]) for i in range(n)] for j in range(3)]
Win = np.reshape(Win, (n,3))

W_r = random_adjacency_matrix(n)
# Wout = np.random.random((3, n)) * 100

W_r = (1 / np.max(np.linalg.eigvals(W_r))) * W_r

l = 0.5
Wout_set = []

s_set = []
c = 0.5
for i in range(50):
    Wout = [[random.choice([0, -c, c]) for i in range(3)] for j in range(n)]
    Wout = np.reshape(Wout, (3,n))
    s = 0
    for k in range(washout):
        r = (1 - l) * r + l * np.arctan(np.reshape(Win @ np.array([x[k], y[k], z[k]]), (n, 1)) + W_r @ r)
        s = s + np.linalg.norm(np.array([x[k], y[k], z[k]]) - Wout @ r, None)

    s_set.append(s)
    Wout_set.append(Wout)

for i in range(20):
    if s_set[i] == min(s_set):
        Wout = Wout_set[i]
    else:
        None

# print(Wout)

for k in range(washout + training):
    r = (1 - l) * r + l * np.arctan(np.reshape(Win @ np.array([x[k], y[k], z[k]]), (n, 1)) + W_r @ r)

x, y, z = [], [], []

for k in range(prediction):
    r = (1 - l) * r + l * np.arctan(np.reshape(Win @ (Wout @ r), (n, 1)) + W_r @ r)
    x.append(float((Wout @ r)[0]))
    y.append(float((Wout @ r)[1]))
    z.append(float((Wout @ r)[2]))



plt.figure('lorenz prediction')

ax = plt.axes(projection='3d')

# Data for a three-dimensional line
xline = x
yline = y
zline = z
ax.plot3D(xline, yline, zline, 'blue')

plt.show()
