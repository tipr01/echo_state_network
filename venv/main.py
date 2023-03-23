import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import space as spc

#reservoir size
n = 100

#timedomain
tmax = 20

#values of numbers of washout-, training -and prediction phase
washout = 500
training = 1000
prediction_time = 20 #time unit


#leaking rate
l = 0.3

#spectral radius of W_r
lmd = 10

#density of W_r
density = 0.1

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
ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.plot3D(*data, 'blue', linewidth=linewidth)
ax.plot(*crit_pnt1, 'fuchsia', marker='o', markersize=2)
ax.plot(*crit_pnt2, 'fuchsia', marker='o', markersize=2)


ax.set_title('solution')

#extend solution
xnew = data[:, -1]

print('x')
t = np.linspace(0, prediction_time, prediction + 1)
sol = solve_ivp(spc.lorenz, (0, prediction_time), xnew, method='RK45', t_eval=t, args=(a, b, c))
print('x')
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
    # Wout = np.linalg.lstsq(X.T, Xtarget.T, rcond=None)[0]
    # Wout = Wout.T

    print(np.linalg.norm(Wout @ X - Xtarget))
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

    # prediction matrix in which columns the predicted coordinates are contained
    X_pred = Wout @ R

    # computation of the error with respect to the right trajectory (2-norm)
    error = []
    nmse = 0
    for i in range(prediction):
        diff = np.linalg.norm(X_pred[:, i] - coo[:, i])
        nmse = nmse + ((diff ** 2) / np.linalg.norm(coo[:, i]) ** 2 )
        error.append(diff)

    nmse = (1 / prediction) * nmse

    # checking the error
    norm = max(np.linalg.norm(X_pred, axis=0))

    # integrating testfunction with respect to the measure...
    normal = crit_pnt2 - crit_pnt1
    point_at_surface = np.array([0, 0, b - 1])
    pas = point_at_surface
    const = 0.5
    iterable1 = [spc.test_func(X_pred[:, i], pas, normal, const)  for i in  range(prediction)]
    iterable2 = [spc.test_func(coo[:, i], pas, normal, const) for i in range(prediction)]

    integral1 = 1 / prediction * np.sum(iterable1)
    integral2 = 1 / prediction * np.sum(iterable2)

    print(np.mean(integral1), np.mean(integral2))



print(f'- normalized mean squared error: {nmse}')
print(f'- number of tries: {count}')
print(f'- integral of the testfunction wrt mu, nu: {integral1, integral2}')




# Data for a three-dimensional line
xline = X_pred[0]
yline = X_pred[1]
zline = X_pred[2]

#plot of the lorenz system prediction
ax.plot3D(xline, yline, zline, 'red', label='esn prediction', linewidth=linewidth)

ax.plot(*crit_pnt1, 'fuchsia', marker='o', markersize=2)
ax.plot(*crit_pnt2, 'fuchsia', marker='o', markersize=2)

point  = np.array([0, 0, b - 1])
normal = crit_pnt2 - crit_pnt1
d = point.dot(normal)

# create y and z
y, z = np.meshgrid(range(-20, 20), range(60))

# calculate corresponding x
x = -(normal[1] * y + normal[2] * z - d) * 1 / normal[0]

# plot the surface
ax.plot_surface(x, y, z, alpha=0.4, color='blue')


plt.legend()

fig2 = plt.figure('additional information')

ax = fig2.add_subplot(2, 2, 1)
ax.set_title('test function')

tstf = [spc.test_func(X_pred[:, i], pas, normal, const) for i in range(prediction)]
#wasserstein = [wasserstein_distance(X_pred[:, i], coo[:, i]) for i in range(prediction)]

ax.plot(time, tstf, label='test function')
#ax.plot(time, wasserstein, label='wasserstein distance')

# plt.legend()


ax = fig2.add_subplot(2, 2, 2)

ax.set_title('error wrt time (2-norm)')
ax.plot(time, error, label='error')

ax = fig2.add_subplot(2, 1, 2)
ax.set_title('individual trajectories')

ax.plot(time, coo[0], color='r', label='x')
ax.plot(time, coo[1], color='g', label='y')
ax.plot(time, coo[2], color='b', label='z')

ax.plot(time, xline, 'r--', label='x_pred')
ax.plot(time, yline, 'g--', label='y_pred')
ax.plot(time, zline, 'b--', label='z_pred')

plt.legend()


plt.show()