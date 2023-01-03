import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#reservoir size
n = 300

#timedomain
tmax = 100

#values of numbers of washout-, training -and prediction phase
washout = 500
training = 100000
prediction_time = 10 #time unit

#leaking rate
l = 0.9


def act(x):
    return np.tanh(x)

#function that creates a random adjacency matrix with spectral radius less than 1
def random_adjacency_matrix(n):
    # matrix = np.random.choice([-1, 0, 1], size=(n, n), p=[0, 0.9, 0.1])
    matrix = np.random.uniform(-1, 1, size=(n, n))

    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0

    for i in range(n):
        for j in range(n):
            if np.random.random() < 0.9:
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

    matrix = np.reshape(matrix, (n,n))
    # set spectral radius to a constant near 1, less than 1 ensures the echo state property
    matrix = (1.0 / np.max(np.linalg.eigvals(matrix))) * matrix
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


steps = washout + training
prediction = int((steps / tmax) * prediction_time)

#timeaxis
t = np.linspace(0, tmax, steps)

#data creation
sol = solve_ivp(lorenz, (0, tmax), x0, method='RK45', t_eval=t, args=(a, b, c))

#plot of the lorenz attractor
time = np.array(sol.t)
coo = np.array(sol.y)
x, y, z = coo[0], coo[1], coo[2]

x, y, z = np.array(x), np.array(y), np.array(z)

data = np.array([x, y, z])
print(np.shape(data[:, 0]))

#extend solution
xnew = np.array([x[-1], y[-1], z[-1]])

t = np.linspace(0, prediction_time, prediction)
sol = solve_ivp(lorenz, (0, prediction_time), xnew, method='RK45', t_eval=t, args=(a, b, c))

time = np.array(sol.t)
coo = np.array(sol.y)
x, y, z = coo[0], coo[1], coo[2]

#
# plt.plot(t, x, color='r', label='x')
# plt.plot(t, y, color='g', label='y')
# plt.plot(t, z, color='b', label='z')
#
# plt.show()


plt.figure('lorenz solution')
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
ax.plot3D(x, y, z, 'blue')


#implementation of the echo state network

#initial value of the reservoir
r = np.zeros(n)

#random (input-) matrix that maps the data for every timestep into the reservoir
#Win = (np.random.rand(n, 3) - 0.5) * 2
Win = np.random.uniform(-1, 1, size=(n, 3))
#Win = np.array([np.identity(3) for i in range(n // 3)]).reshape((n, 3))


for i in range(n):
    for j in range(3):
        if np.random.random() < 2/3:
            Win[i][j] = 0

#random adjacency matrix which connects some of the "reservoir points"
W_r = random_adjacency_matrix(n)


for k in range(washout):
    r = (1 - l) * r + l * act(Win @ data[:, k] + W_r @ r)



#creates matrix of reservoir states in training phase
X = []
R = []
print('Finding reservoir states...')
perc = 0

for k in range(washout, steps):
    if (k * 100) // training  > perc:
        perc += 1
        if perc % 10 == 0:
            print(f"{perc // 10}", sep='', end='', flush=True)
        else:
            print("▮", sep='', end='', flush=True)
        if perc == 99:
            print('')

    r = (1 - l) * r + l * act(Win @ data[:, k] + W_r @ r)
    X.append(r)

    if k in range(steps-3, steps):
        R.append(r)

print('Completed.')
# plt.figure()
# plt.plot([i for i in range(steps)], X)
# plt.show()

print(np.shape(X))

X = np.array(X).T

#matrix of solutions in training phase
Xtarget = data[:, washout:steps]


#computation of the output matrix via ridge regression with regularization term beta
beta = 1e-8
Wout = Xtarget @ X.T @ np.linalg.inv(X @ X.T + beta * np.identity(n))

# #computation of the error
# s = 0
# r = np.zeros(n)
# for k in range(washout):
#     r = (1 - l) * r + l * np.arctan(Win @ data[:, k] + W_r @ r)
#     s = s + np.linalg.norm(data[:, k] - (Wout @ r), None) ** 2
# print(s)
#
# r = np.zeros(n)
#
# for k in range(training):
#     r = (1 - l) * r + l * np.arctan(Win @ data[:, k] + W_r @ r)
#     #print(Wout @ r)


print('generating prediction...')
perc = 0
for k in range(prediction):
    if (k * 100) // prediction  > perc:
        perc += 1
        if perc % 10 == 0:
            print("|", sep='', end='', flush=True)
        else:
            print("▮", sep='', end='', flush=True)
        if perc == 99:
            print('')

    r = (1 - l) * R[-1] + l * act(Win @ (Wout @ R[-3]) + W_r @ R[-1])
    #print(Wout @ r)
    R.append(r)

print('end prediction')
R = np.array(R).T

X_pred = Wout @ R


# for i in range(prediction):
#     print(X_pred[i])
[print(col) for col in X_pred.T]



# plt.figure('lorenz prediction')

# ax = plt.axes(projection='3d')

# Data for a three-dimensional line
xline = X_pred[0]
yline = X_pred[1]
zline = X_pred[2]
ax.plot3D(xline, yline, zline, 'red')

plt.show()
