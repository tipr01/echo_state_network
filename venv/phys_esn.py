import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import space as spc



#reservoir size
n = 100

#timedomain
tmax = 50

#values of numbers of washout-, training -and prediction phase
washout = 500
training = 1000
prediction_time = 10 #time unit

#constant K
K = 0.1

# distance of steps we want to use from the past
m = n

#lorenz system parameters
a, b, c = 10.0, 30.0, 2

#critical points
crit_pnt1 = np.array([np.sqrt(c * (b - 1)), np.sqrt(c * (b - 1)), b - 1])
crit_pnt2 = np.array([ - np.sqrt(c * (b - 1)), - np.sqrt(c * (b - 1)), b - 1])

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

#print(data)


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
c1, c2, c3 = coo

#plot of the extended solution
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_title('prediction')
ax.plot3D(*coo, 'green', label='solve_ivp solution', linewidth=linewidth)

norm = 101
count = 0

while norm > 100 and count < 100:
    count += 1

    map = np.random.uniform(-0.2, 0.2, size=(n, 3))
    for i in range(n):
        for j in range(3):
            if np.random.random() < 2/3:
                map[i][j] = 0

    reservoir = np.zeros(n) # np.random.uniform(0, 1, size=n)

    j = 1
    for i in range(n, n * washout):
        if i // n > j:
            j += 1
        k = i % n
        reservoir = np.append(reservoir, spc.nonlin(K * reservoir[i - m] + data[:, j].T @ map[k, :]))

    for i in range(n * washout, n * steps):
        if i // n > j:
            j += 1
        k = i % n
        reservoir = np.append(reservoir, spc.nonlin(K * reservoir[i - m] + data[:, j].T @ map[k, :]))

    reservoir_state_matrix = np.reshape(reservoir[n * washout: n * steps], (training, n)).T


    target = data[:, washout:steps]

    Wout_x = np.linalg.lstsq(reservoir_state_matrix.T, target.T, rcond=None)[0].T

    # computation of the output matrix via Tikhonov regularization with regularization term beta (worse than numpy)
    # beta = 1e-8
    # Wout_x = target @ reservoir_state_matrix.T @ np.linalg.inv(reservoir_state_matrix @ reservoir_state_matrix.T + beta * np.identity(n))


    print(np.linalg.norm(Wout_x @ reservoir_state_matrix - target))

    #print(np.shape(Wout_x))
    res_vec = reservoir_state_matrix.T[-1]
    #print(reservoir_state_matrix, res_vec)
    R = [res_vec]
    k = 1
    print(np.shape(coo))
    for j in range(steps, steps + prediction):
        print(Wout_x @ R[-1])
        res_vec = np.empty(n)
        d = Wout_x @ np.array(R[-1])
        for i in range(n):
            if i - m < 0:
                res_vec[i] = spc.nonlin(K * R[-1][n + (i - m)] + d @ map[i, :])
                #print(res_vec[i])
            else:
                res_vec[i] = spc.nonlin(K * res_vec[i - m] + d @ map[i, :])
                #print(res_vec[i])
        k += 1
        R.append(res_vec)

    #print(R)
    R = np.array(R).T

    #reservoir_state_matrix = np.reshape(reservoir[n * steps: n * (steps + prediction)], (prediction, n)).T


    x = Wout_x @ R

    norm = max(np.linalg.norm(x, axis=0))


#plot of the lorenz system prediction
ax.plot3D(*x, 'violet', label='esn prediction', linewidth=linewidth)

ax.plot(*crit_pnt1, 'fuchsia', marker='o', markersize=2)
ax.plot(*crit_pnt2, 'fuchsia', marker='o', markersize=2)

plt.legend()
plt.show()