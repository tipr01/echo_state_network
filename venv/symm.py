import numpy as np

#
# n = 4
# m = 3
# p = 3
#
# for i in range(100):
#     G = np.random.rand(n, m)
#     B = np.random.rand(n, p)
#     C = np.random.rand(m, p)
#
#     R = np.random.rand(m, m)
#     R = R @ R.T
#
#     #print(R)
#     J = np.random.rand(m, m)
#     J = J - J.T
#     #print(J)
#     inv = np.linalg.inv(np.identity(m) - J @ R)
#     #print(R @ inv, '\n', inv @ R)
#     P = B -  G @ R @ inv @ C
#     Ps = B.T - C.T @ R @ inv @ G.T
#     print(np.round(P, 1), '\n', np.round(Ps, 1).T)
#
#     if (np.round(P) == np.round(Ps).T).all():
#         print('jes')
#     else:
#         print('no')


