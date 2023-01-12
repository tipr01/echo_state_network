import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import space as spc



#reservoir size
n = 300

#timedomain
tmax = 20

#values of numbers of washout-, training -and prediction phase
washout = 200
training = 500
prediction_time = 1 #time unit

#constant K
K = 1

# distance of steps we want to use from the past
m = n - 1

#lorenz system parameters
a, b, c = 10.0, 30.0, 2

#critical points
crit_pnt1 = np.array([np.sqrt(c * (b - 1)), np.sqrt(c * (b - 1)), b-1])
crit_pnt2 = np.array([ - np.sqrt(c * (b - 1)), - np.sqrt(c * (b - 1)), b-1])

#initial value
x0 = np.array([1.0, 0.0, 0.0])

# computation of stepsizes
steps = washout + training
prediction = int((steps / tmax) * prediction_time)

#timeaxis
t = np.linspace(0, tmax, steps)

#data creation
sol = solve_ivp(spc.lorenz, (0, tmax), x0, method='RK45', t_eval=t, args=(a, b, c))
time = np.array(sol.t)
data = np.array(sol.y)
x, y, z = data


#plot of the lorenz attractor
linewidth = 0.8

fig = plt.figure('lorenz system prediction', figsize=plt.figaspect(0.5))
# manager =  plt.get_current_fig_manager()
ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.plot3D(*data, 'blue', linewidth=linewidth)
ax.plot(*crit_pnt1, 'fuchsia', marker='o', markersize=2)
ax.plot(*crit_pnt2, 'fuchsia', marker='o', markersize=2)


ax.set_title('solution')

#extend solution
xnew = data[:, -1]

t = np.linspace(0, prediction_time, prediction + 1)
sol = solve_ivp(spc.lorenz, (0, prediction_time), xnew, method='RK45', t_eval=t, args=(a, b, c))

time = np.array(sol.t)[1:]
coo = np.array(sol.y)[:, 1:]

#plot of the extended solution
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_title('prediction')
ax.plot3D(*coo, 'green', label='solve_ivp solution', linewidth=linewidth)

X_wash = np.empty((n, washout))
Y_wash = np.empty((n, washout))
Z_wash = np.empty((n, washout))

map = np.random.uniform(1, 10, size=n)

reservoir_iv = np.ones(n) # np.random.uniform(0, 1, size=n)
X_wash[:, 0] = reservoir_iv
Y_wash[:, 0] = reservoir_iv
Z_wash[:, 0] = reservoir_iv

for i in range(1, washout):
    for k in range(n):
        X_wash[k, i] = spc.nonlin(K * X_wash[k - m, i] + x[i] * map[k])
        Y_wash[k, i] = spc.nonlin(K * Y_wash[k - m, i] + y[i] * map[k])
        Z_wash[k, i] = spc.nonlin(K * Z_wash[k - m, i] + z[i] * map[k])

X = np.zeros((n, training))
Y = np.zeros((n, training))
Z = np.zeros((n, training))

X[:, 0] = X_wash[:, -1]
Y[:, 0] = Y_wash[:, -1]
Z[:, 0] = Z_wash[:, -1]

for i in range(1, training):
    for k in range(n):
        X[k, i] = spc.nonlin(K * X[k - m, i] + x[i] * map[k])
        Y[k, i] = spc.nonlin(K * Y[k - m, i] + y[i] * map[k])
        Z[k, i] = spc.nonlin(K * Z[k - m, i] + z[i] * map[k])

x_target = data[0, washout:steps]
y_target = data[1, washout:steps]
z_target = data[2, washout:steps]


Wout_x = np.linalg.lstsq(X.T, x_target.T, rcond=None)[0].T
Wout_y = np.linalg.lstsq(Y.T, y_target.T, rcond=None)[0].T
Wout_z = np.linalg.lstsq(Z.T, z_target.T, rcond=None)[0].T

print(np.shape(Wout_x))

X_pred = np.zeros((n, prediction))
Y_pred = np.zeros((n, prediction))
Z_pred = np.zeros((n, prediction))

X_pred[:, 0] = X[:, -1]
Y_pred[:, 0] = Y[:, -1]
Z_pred[:, 0] = Z[:, -1]


for i in range(1, prediction):
    for k in range(n):
        X_pred[k, i] = spc.nonlin(K * X_pred[k - m, i] + Wout_x @ X_pred[:, i - 1] * map[k])
        Y_pred[k, i] = spc.nonlin(K * Y_pred[k - m, i] + Wout_y @ Y_pred[:, i - 1] * map[k])
        Z_pred[k, i] = spc.nonlin(K * Z_pred[k - m, i] + Wout_z @ Z_pred[:, i - 1] * map[k])


coo = Wout_x @ X_pred, Wout_y @ Y_pred, Wout_z @ Z_pred

#plot of the lorenz system prediction
ax.plot3D(*coo, 'red', label='esn prediction', linewidth=linewidth)

ax.plot(*crit_pnt1, 'fuchsia', marker='o', markersize=2)
ax.plot(*crit_pnt2, 'fuchsia', marker='o', markersize=2)


plt.legend()

plt.show()

plt.figure()
plt.plot(time, coo[0])
plt.plot(time, coo[1])
plt.plot(time, coo[2])



plt.show()

