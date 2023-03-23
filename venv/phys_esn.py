import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import space as spc



#reservoir size
n = 100

#timedomain
tmax = 22

#values of numbers of washout-, training -and prediction phase
washout = 500
training = 1000
prediction_time = 5 #time unit

#constant K
K = 1

# distance of steps we want to use from the past
m = n // 2

#lorenz system parameters
a, b, c = 10.0, 30.0, 2

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
ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.plot3D(*data, 'blue', linewidth=linewidth)
ax.plot(*spc.crit_pnt1, 'fuchsia', marker='o', markersize=2)
ax.plot(*spc.crit_pnt2, 'fuchsia', marker='o', markersize=2)

ax.set_title('solution')

#extend solution
xnew = data[:, -1]

t = np.linspace(0, prediction_time, prediction + 1)
sol = solve_ivp(spc.lorenz, (0, prediction_time), xnew, method='RK45', t_eval=t, args=(a, b, c))

time = np.array(sol.t)[1:]
coo = np.array(sol.y)[:, 1:]
c1, c2, c3 = coo

#plot of the extended solution
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_title('prediction')
ax.plot3D(*coo, 'green', label='solve_ivp solution', linewidth=linewidth)

norm = 101
count = 0

while norm > 100 and count < 100:
    count += 1

    map = np.random.uniform(-0.02, 0.02, size=(n, 3))
    print(np.shape(map[2, :]))
    reservoir = np.random.uniform(0, 1, size=n)
    reservoir, j = spc.washout(reservoir, washout, n, m, data, map, K, 1)
    reservoir_state_matrix = spc.training(reservoir, washout, steps, n, m, data, map, K, j)
    target = data[:, washout:steps]

    Wout_x = np.linalg.lstsq(reservoir_state_matrix.T, target.T, rcond=None)[0].T

    # computation of the output matrix via Tikhonov regularization with regularization term beta (worse than numpy)
    # beta = 1e-8
    # Wout_x = target @ reservoir_state_matrix.T @ np.linalg.inv(reservoir_state_matrix @ reservoir_state_matrix.T + beta * np.identity(n))


    print('norm = ', np.linalg.norm(Wout_x @ reservoir_state_matrix - target))

    res_vec = reservoir_state_matrix.T[-1]
    x = spc.prediction([], n, m, steps, prediction, 1, res_vec, Wout_x, K, map)

    norm = max(np.linalg.norm(x, axis=0))


#plot of the lorenz system prediction
ax.plot3D(*x, 'violet', label='esn prediction', linewidth=linewidth)

ax.plot(*spc.crit_pnt1, 'fuchsia', marker='o', markersize=2)
ax.plot(*spc.crit_pnt2, 'fuchsia', marker='o', markersize=2)
ax.plot(*x[0], 'red', marker='o', markersize=2)
ax.plot(*x[-1], 'black', marker='o', markersize=2)


plt.legend()
plt.show()