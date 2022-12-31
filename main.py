import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


#function that creates a random adjacency matrix with spectral radius less than 1
def random_adjacency_matrix(n):
    matrix = np.random.choice([-1, 0, 1], size=(n, n), p=[0.1, 0.8, 0.1])
            #np.random.uniform(-1, 1, size=(n, n))

    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0

    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            if np.abs(matrix[j][i]) == 1 or np.abs(matrix[i][j]) == 1:
                if (i % 2) == 0:
                    matrix[i][j], matrix[j, i] = 1, 1
                else:
                    matrix[i][j], matrix[j, i] = -1, -1
            else:
                matrix[i][j] = matrix[j][i]

    matrix = np.reshape(matrix, (n,n))
    matrix = (1 / np.max(np.linalg.eigvals(matrix))) * matrix
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

#initial value
x0 = np.array([1.0, 0.0, 0.0])

#timedomain
tmax = 20

#values of numbers of washout-, training -and prediction phase
washout = 500
training = 10000
prediction = 500

steps = washout + training

#timeaxis
t = np.linspace(0, tmax, steps)

#data creation
sol = solve_ivp(lorenz, (0, tmax), x0, method='RK45', t_eval=t, args=(a, b, c))

#plot of the lorenz attractor
time = np.array(sol.t)
coo = np.array(sol.y)
x, y, z = coo[0], coo[1], coo[2]

plt.figure('lorenz solution')
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
ax.plot3D(x, y, z, 'blue')


#implementation of the echo state network

#reservoir size
n = 100

#initial value of the reservoir
r = np.zeros(n)

#random (input-) matrix that maps the data for every timestep into the reservoir
Win = (np.random.rand(n,3) - 0.5) * 2

#random adjacency matrix which connects some of the "reservoir points"
W_r = random_adjacency_matrix(n)

#leaking rate
l = 0.8

#creates matrix of reservoir states in training phase
X = []
for k in range(training):
    r = (1 - l) * r + l * np.arctan(Win @ np.array([x[k], y[k], z[k]]) + W_r @ r)
    X.append(r)

X = np.array(X).reshape((n, training))

#matrix of solutions in training phase
Xtarget = [np.array([x[k], y[k], z[k]]) for k in range(training)]
Xtarget = np.array(Xtarget).T


#computation of the output matrix via ridge regression with regularization term beta
beta = 1e-8
Wout = Xtarget @ X.T @ np.linalg.inv(X @ X.T + beta * np.identity(n))

#computation of the error
# s = 0
# for k in range(washout):
#     r = (1 - l) * r + l * np.arctan(Win @ np.array([x[k], y[k], z[k]]) + W_r @ r)
#     s = s + np.linalg.norm(np.array([x[k], y[k], z[k]]) - (Wout @ r), None)
# print(s)

r = np.zeros(n)

for k in range(training):
    r = (1 - l) * r + l * np.arctan(Win @ np.array([x[k], y[k], z[k]]) + W_r @ r)
    print(Wout @ r)



R = []
for k in range(prediction):
    r = (1 - l) * r + l *  np.arctan(Win @ (Wout @ r) + W_r @ r)
    #print(Wout @ r)
    R.append(r)

X_pred = Wout @ np.array(R).reshape((n, prediction))




plt.figure('lorenz prediction')

ax = plt.axes(projection='3d')

# Data for a three-dimensional line
xline = X_pred.T[0]
yline = X_pred.T[1]
zline = X_pred.T[2]
ax.plot3D(xline, yline, zline, 'blue')

#plt.show()
