import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random
import scipy.io
import phot_rc as ph


training = 10000
prediction = 100


#mask size
Nv = 30

dt = 0.02
#timedomain
tmax = int(training * dt)

#time series stepsize
stepsize = 1e-3
steps = int(tmax / stepsize)

cut = int(steps * 0.1)

#constant K
K = 0.02

# delay
m = Nv + 1

#input scaling
G = 0.3

data = scipy.io.loadmat('coordinate_data.mat')
data = np.array(data['out'])

#timeaxis
time = np.linspace(0, tmax, steps)

time = time[cut:]
x, y, z = data[:, cut:]
#print(np.shape(time)[0], np.shape(x)[0])


# standardization of data
minx, miny, minz = np.min(x), np.min(y), np.min(z)
difx = np.max(x) - minx
dify = np.max(y) - miny
difz = np.max(z) - minz

x = (x - minx) / difx
y = (y - miny) / dify
z = (z - minz) / difz


mask = np.random.uniform(0, 1, size=Nv) # np.array([0 if np.random.random() < 0.5 else 1 for i in range(Nv)]) #np.random.uniform(0, 1, size=Nv)
#print(mask)
mask3 = np.random.uniform(0, 1, size=(Nv, 3))


indices = [int(x.shape[0] / training * i) for i in range(training)]
xnew = np.take(x, indices)
ynew = np.take(y, indices)
znew = np.take(z, indices)

data_mat = np.array([[xnew[i], ynew[i], znew[i]] for i in range(xnew.shape[0])]).T

print(data_mat.shape[1])

# plt.plot(np.take(time, indices), xnew)
# plt.show()

c = 1
S = ph.state_mat(xnew[:-c], mask, m, G)
S3 = ph.state_mat_multi_data(data_mat[:, :-c], mask3, m, G)

# computation of the output matrix via Tikhonov regularization with regularization term beta
xtarg, ytarg, ztarg = xnew[c:], ynew[c:], znew[c:]

Wx_out, Wy_out, Wz_out = ph.tikhonov(S3, xtarg), ph.tikhonov(S, ytarg), ph.tikhonov(S, ztarg)
Wout = ph.tikhonov(S3, data_mat[:, c:].T)  #Wx_out, Wy_out, Wz_out
#Wout = np.array(Wout).T

print('norm = ', np.linalg.norm(S @ Wx_out - xnew[c:].T))

pred = scipy.io.loadmat('coordinate_pred.mat')
pred = np.array(pred['out'])

predx = np.take(pred[0, :], indices)
predx = predx[:prediction]
predx = (predx - minx) / difx


init_vec = S[-1, :]
initialization = xnew[-1] # predx[0] # xnew[-1]

S_auto_reg = ph.state_mat_autoregr(initialization, init_vec, Wout[:, 0], mask, m, prediction, G, predx)
#print(S_auto_reg[-1, :])

# for i in range(-5, 0):
#     print(i,':  ',  S[i, :] @ Wx_out, xnew[i])
#
# for i in range(5):
#     print(i,':   ', S_auto_reg[i, :] @ Wx_out, predx[i])

pred_auto_x = (S_auto_reg @ Wout[:, 0]).T

print('NRMSE x-to-x phot: %5.5f\n' % ph.nrmse(pred_auto_x[:], predx))
# print('NRMSE x-to-y phot: %5.5f\n' % nrmse(pred[:], c2))
# print('NRMSE x-to-z phot: %5.5f\n' % nrmse(pred[:], c3))

pred_axis = np.linspace(0, tmax, steps)
pred_axis = np.take(pred_axis, indices)

plt.plot(pred_axis[:prediction], predx * difx + minx, label='sol')
plt.plot(pred_axis[:prediction], pred_auto_x * difx + minx, label='pred')

plt.legend()
plt.show()