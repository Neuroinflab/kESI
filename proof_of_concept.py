import numpy as np
import matplotlib.pyplot as plt

NODES = np.array([-1., 0., 1.])
X = np.linspace(-5, 5, 1000)

F = {'1': lambda X: np.ones_like(X),
     'X': lambda X: X,
     'X**2': lambda X: X ** 2,
     }
F_PRIM = {'1': lambda X: np.zeros_like(X),
          'X': lambda X: np.ones_like(X),
          'X**2': lambda X: 2 * X,
          }
 

def getCrossKernel(fs1, fs2, NODES, X):
    return np.matrix([[sum(f1(x) * f2(y) for f1, f2 in zip(fs1, fs2))
                       for x in NODES
                       ]
                      for y in X
                      ])


FS = [f for _, f in sorted(F.items())]
FS_PRIM = [f for _, f in sorted(F_PRIM.items())]

K_INV = np.linalg.inv(getCrossKernel(FS, FS, NODES, NODES))
K_F = getCrossKernel(FS, FS, NODES, X)
K_F_PRIM = getCrossKernel(FS, FS_PRIM, NODES, X)

INT_F = K_F * K_INV
INT_F_PRIM = K_F_PRIM * K_INV

for (x0, x1, x2) in [(0, 1, 1),
                     (1, 0, 0),
                     (0, 0, 0),
                     ]:
    f = lambda x: (x2 * x + x1) * x + x0
    f_prim = lambda x: 2 * x2 * x + x1
    Y_NODES = np.array(f(NODES)).reshape(-1, 1)

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    fig.suptitle('${2} x^2 + {1} x + {0}$'.format(x0, x1, x2))
    ax1.plot(X, f(X)) 
    ax1.plot(X, INT_F * Y_NODES, ls='--') 
    ax2.plot(X, f_prim(X)) 
    ax2.plot(X, INT_F_PRIM * Y_NODES, ls='--')

plt.show()
