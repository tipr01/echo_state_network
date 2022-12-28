import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



def random_adjacency_matrix(n):
    matrix = np.random.choice([-1, 0, 1], size=(n, n), p=[0.05, 0.9, 0.05])
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
tmax = 50

washout = 500
training = 10000
prediction = 1000

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



n = 60


r = np.ones(n) #np.random.rand(n, 1) * 100
#r = np.array(r)

# Win = np.random.random((n, 3)) * 100
# Win = [[random.choice([0, -c, c]) for i in range(n)] for j in range(3)]
# Win = np.reshape(Win, (n,3))

Win = np.random.choice([0, 1], size=(n, 3), p=[0.7, 0.3]) #scs.random(n, 3, density=0.25, random_state=None) * 10
W_r = random_adjacency_matrix(n)
W_r = (1 / np.max(np.linalg.eigvals(W_r))) * W_r

l = 1.0

X = []

for k in range(washout + training):
    r = (1 - l) * r + l * np.arctan(Win @ np.array([x[k], y[k], z[k]]) + W_r @ r)
    X.append(r)

X = np.array(X).reshape((n, washout + training))

beta = 10
Xtarget = [np.array([x[k], y[k], z[k]]) for k in range(washout + training)]
Xtarget = np.array(Xtarget).T

Wout = Xtarget @ X.T @ np.linalg.inv(X @ X.T + beta * np.identity(n))

s = 0
for k in range(washout):
    r = (1 - l) * r + l * np.arctan(Win @ np.array([x[k], y[k], z[k]]) + W_r @ r)
    s = s + np.linalg.norm(np.array([x[k], y[k], z[k]]) - (Wout @ r), None)
print(s)

x, y, z = [], [], []

for k in range(prediction):
    r = (1 - l) * r + l * np.arctan(Win @ (Wout @ r) + W_r @ r)
    x.append((Wout @ r)[0])
    y.append((Wout @ r)[1])
    z.append((Wout @ r)[2])



plt.figure('lorenz prediction')

ax = plt.axes(projection='3d')

# Data for a three-dimensional line
xline = x
yline = y
zline = z
ax.plot3D(xline, yline, zline, 'blue')

plt.show()
