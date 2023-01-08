import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import space as spc

#reservoir size
n = 100

#timedomain
tmax = 20

#values of numbers of washout-, training -and prediction phase
washout = 500
training = 1000
prediction_time = 10 #time unit

#leaking rate
l = 0.9

#spectral radius of W_r
lmd = 10

#density of W_r
density = 0.1

#lorenz system parameters
a, b, c = 10.0, 30.0, 2

#critical points

crit_pnt1 = np.array([np.sqrt(c * (b - 1)), np.sqrt(c * (b - 1)), b-1])
crit_pnt2 = np.array([ - np.sqrt(c * (b - 1)), - np.sqrt(c * (b - 1)), b-1])
# crit_pnt1 = str(crit_pnt1)
# crit_pnt2 = str(crit_pnt2)

# print(crit_pnt1, crit_pnt2)

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


#implementation of the echo state network
norm = 101
X_pred = 0
count = 0

# Due to the random choice of the matrices, the prediction in the "far" future fails more often,
# so the whole ESN is implemented in a while loop that checks for the error.
while norm > 100 and count < 100:
    count += 1

    #initial value of the reservoir
    r = np.zeros(n)

    #random (input-) matrix that maps the data for every timestep into the reservoir

    #Win = (np.random.rand(n, 3) - 0.5) * 2
    Win = np.random.uniform(-0.1, 0.1, size=(n, 3))
    #Win = np.array([np.identity(3) for i in range(n // 3)]).reshape((n, 3))
    for i in range(n):
        for j in range(3):
            if np.random.random() < 2/3:
                Win[i][j] = 0

    #random adjacency matrix which connects some of the "reservoir points"
    W_r = spc.random_adjacency_matrix(n, density, lmd)

    for k in range(washout):
        r = (1 - l) * r + l * spc.act(Win @ data[:, k] + W_r @ r)

    #creates matrix of reservoir states in training phase
    X = []
    print('Computing reservoir states in training phase...')
    perc = 0

    for k in range(washout, steps):
        if ((k - washout) * 100) // training  > perc:
            perc += 1
            if perc % 10 == 0:
                print(f"{perc // 10}", sep='', end='', flush=True)
            else:
                print("▮", sep='', end='', flush=True)
            if perc == 99:
                print('')
        X.append(r)
        r = (1 - l) * r + l * spc.act(Win @ data[:, k] + W_r @ r)
    print('Completed.')

    X = np.array(X).T

    #matrix of solutions in training phase
    Xtarget = data[:, washout:steps]

    #computation of the output matrix via Tikhonov regularization with regularization term beta (worse than numpy)
    beta = 1e-8
    Wout = Xtarget @ X.T @ np.linalg.inv(X @ X.T + beta * np.identity(n))

    # computation of the output matrix via numpy least squares
    #Wout = np.linalg.lstsq(X.T, Xtarget.T, rcond=None)[0]
    #Wout = Wout.T

    #print(np.linalg.norm(Wout @ X - Xtarget))
    print('Generating prediction...')

    R = []
    perc = 0
    for k in range(prediction):
        if (k * 100) // prediction  > perc:
            perc += 1
            if perc % 10 == 0:
                print(f"{perc // 10}", sep='', end='', flush=True)
            else:
                print("▮", sep='', end='', flush=True)
            if perc == 99:
                print('')

        # r = (1 - l) * r + l * spc.act(Win @ data[:, k] + W_r @ r)
        #print(data[:, k], Wout @ r)
        R.append(r)
        r = (1 - l) * r + l * spc.act(Win @ (Wout @ r) + W_r @ r)
        #print(Wout @ r)

    print('Prediction completed.')

    R = np.array(R).T

    X_pred = Wout @ R


    error = []
    for i in range(prediction):
        error.append(np.linalg.norm(X_pred[:, i] - coo[:, i]))


    frechet_var1 = np.sum([np.linalg.norm(crit_pnt1 - X_pred.T[k]) ** 2 for k in range(prediction)])
    frechet_var2 = np.sum([np.linalg.norm(crit_pnt2 - X_pred.T[k]) ** 2 for k in range(prediction)])

    print(frechet_var1)
    print(frechet_var2)

    norm = max(np.linalg.norm(X_pred, axis=0))

print(f'Number of tries: {count}')

plt.figure('error')
plt.plot(time, error, label='error')





# Data for a three-dimensional line
xline = X_pred[0]
yline = X_pred[1]
zline = X_pred[2]

#plot of the lorenz system prediction
ax.plot3D(xline, yline, zline, 'red', label='esn prediction', linewidth=linewidth)

ax.plot(*crit_pnt1, 'fuchsia', marker='o', markersize=2)
ax.plot(*crit_pnt2, 'fuchsia', marker='o', markersize=2)


plt.legend()
#
# plt.figure('individual trajectories')
#
# plt.plot(time, coo[0], color='r', label='x')
# plt.plot(time, coo[1], color='g', label='y')
# plt.plot(time, coo[2], color='b', label='z')
#
# plt.plot(time, xline, 'r--', label='x_pred', )
# plt.plot(time, yline, 'g--', label='y_pred')
# plt.plot(time, zline, 'b--', label='z_pred')
#
# plt.legend()

plt.show()