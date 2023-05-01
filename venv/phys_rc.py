import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random
import scipy.io
import phot_rc as ph
import pandas
import plotly.express as px




datasize = 35000
step1 = 10000
step2 = 20000
step3 = 30000
step4 = datasize


prediction = 222

#mask size
Nv = 30

dt = 0.02

#timedomain
tmax = int(datasize * dt)

#time series stepsize
stepsize = 1e-3
steps = int(tmax / stepsize)

# delay
m = Nv + 1

print('loooad data')
data = scipy.io.loadmat('coordinate_data.mat')
data = np.array(data['out'])
print('end loooad data')
#timeaxis
time = np.linspace(0, tmax, datasize)

x, y, z = data
cut = 1000

# standardization of data
minx, miny, minz = np.min(x[cut:]), np.min(y[cut:]), np.min(z[cut:])
difx = np.max(x) - minx
dify = np.max(y) - miny
difz = np.max(z) - minz

x = (x - minx) / difx
y = (y - miny) / dify
z = (z - minz) / difz

input_scale_values = []
feedback_strenght_values = []
error = []

for i in range(30):
    print(i)
    for j in range(30):
        # input scaling
        G = 0.01 * (i + 1)
        input_scale_values.append(G)

        K = 0.01 * (j + 1)
        feedback_strenght_values.append(K)

        #np.random.seed(5)
        mask = np.random.uniform(0, 1, size=Nv) # np.array([0 if np.random.random() < 0.5 else 1 for i in range(Nv)]) #np.random.uniform(0, 1, size=Nv)
        #print(mask)
        #mask3 = np.random.uniform(0, 1, size=(Nv, 3))
        # data_mat = np.array([[xnew[i], ynew[i], znew[i]] for i in range(xnew.shape[0])]).T

        S = ph.state_mat(x, mask, m, G, K)
        S_tr = S[step1-1:step2-1, :]
        S_te = S[step3-1:-1, :]

        #S3 = ph.state_mat_multi_data(data_mat[:, :-c], mask3, m, G)

        # computation of the output matrix via Tikhonov regularization with regularization term beta
        c1 = step1
        c2 = step2
        xtarg = x[c1:c2]
        #ytarg = y[c1:c2]
        #ztarg = z[c1:c2]

        Wx_out = ph.tikhonov(S_tr, xtarg)
        # Wy_out = ph.tikhonov(S_tr, ytarg)
        # Wz_out = ph.tikhonov(S_tr, ztarg)
        #Wout = ph.tikhonov(S3, data_mat[:, c:].T)  #Wx_out, Wy_out, Wz_out
        #Wout = np.array(Wout).T

        #print('norm = ', np.linalg.norm(S_tr @ Wx_out - x[c1:c2].T))

        #time = np.linspace((6 / 7) * tmax, tmax, 5000)
        predx = S_te @ Wx_out
        # predz = S_te @ Wz_out
        # print('NRMSE x-to-x phot: %5.5f\n' % ph.nrmse(predx, x[step3: ]))
        # print('NRMSE x-to-z phot: %5.5f\n' % ph.nrmse(predz, z[step3: ]))

        error.append(ph.nrmse(predx, x[step3: ]))
        #print(f'({i}, {j})')
fig = px.scatter(x=feedback_strenght_values, y=input_scale_values, color=error,
                 title="nrmse")
fig.update_traces(marker_size=20)

fig.show()



# plt.plot(time , x[step3:] * difx + minx, label='sol')
# plt.plot(time, predx * difx + minx, label='pred')
#
# plt.legend()
# plt.show()

# pred = scipy.io.loadmat('coordinate_pred.mat')
# pred = np.array(pred['out'])
#
# predx = pred[0, :]
# predx = predx[:prediction]
# predx = (predx - minx) / difx
#
# initialization = x[-1] #predx[0]
# init_vec = S_te[-1, :]
# S_auto_reg = ph.state_mat_autoregr(initialization, init_vec, Wx_out, mask, m, prediction, G)
#
# for i in range(-5, 0):
#     print(i,':  ',  S_te[i, :] @ Wx_out, x[i])
#
# for i in range(5):
#     print(i,':   ', S_auto_reg[i, :] @ Wx_out, predx[i])
#
# pred_auto_x = (S_auto_reg @ Wx_out).T
#
# print('NRMSE x-to-x phot: %5.5f\n' % ph.nrmse(pred_auto_x[:], predx))
# # print('NRMSE x-to-y phot: %5.5f\n' % nrmse(pred[:], c2))
# # print('NRMSE x-to-z phot: %5.5f\n' % nrmse(pred[:], c3))
#
# pred_axis = np.linspace(0, tmax, datasize)
#
# plt.plot(pred_axis[:prediction], predx * difx + minx, label='sol')
# plt.plot(pred_axis[:prediction], pred_auto_x * difx + minx, label='pred')
#
# plt.axis([0, prediction * tmax / datasize, minx - 1, -minx +1])
# plt.legend()
# plt.show()