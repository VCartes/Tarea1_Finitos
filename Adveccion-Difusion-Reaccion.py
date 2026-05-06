from dolfin import *

import sympy as sp
import numpy as np

from prettytable import PrettyTable


# ----- GLOBAL VARIABLES -----

a           = 0
b           = 1
epsilon     = 1e-1
alpha       = 1
beta        = 0
u0          = 0
u1          = 0


x = sp.symbols("x[0]")

f = 1

p = (u1-u0)/(b-a)*(x-a) + u0
u_ex = x - (sp.exp(-(1-x)/epsilon) - sp.exp(-1/epsilon)) / (1 - sp.exp(-1/epsilon))


f = Expression(sp.printing.ccode(f), degree = 5)
u_ex = Expression(sp.printing.ccode(u_ex), degree = 5)


def boundary(x):
    return x[0] < a + DOLFIN_EPS or x[0] > b - DOLFIN_EPS

def L2_norm(f):
    return np.sqrt(assemble(f**2 * dx))

def H1_norm(f, FS):
    f_proj = project(f, FS)
    norm = L2_norm(f_proj)
    norm += L2_norm(Dx(f_proj, 0))
    return norm


def vf_solve(n_interv, deg):
    mesh = IntervalMesh(n_interv, a, b)

    V = FunctionSpace(mesh, "Lagrange", deg)
    
    u_dir = Expression(sp.printing.ccode(p), degree = 5)
    bc = DirichletBC(V, u_dir, boundary)


    # Variational Problem
    u = TrialFunction(V)
    v = TestFunction(V)

    A = epsilon*(Dx(u, 0)*Dx(v, 0)*dx) + alpha*(Dx(u, 0)*v*dx) + beta*(u*v*dx)
    F = f*v*dx


    # Solve problem
    u = Function(V)
    solve(A == F, u, bc)

    # Error
    L2_err = L2_norm(u - u_ex)
    H1_err = H1_norm(u - u_ex, V)

    return u, mesh.hmax(), L2_err, H1_err


# ----- MAIN PROGRAM -----


if __name__ == "__main__":
    degree  = 1
    NN      = [8, 16, 32, 64]

    u       = None
    hh      = list()
    L2_errs = list()
    H1_errs = list()

    for n in NN:
        u, h, L2_err, H1_err = vf_solve(n, degree)

        hh.append(h)
        L2_errs.append(L2_err)
        H1_errs.append(H1_err)


    table = PrettyTable(["h", "err_L2", "err_H1"])
    table.border = True

    table.add_rows([["%.2f" % hh[i], "%2.2e" % L2_errs[i], "%2.2e" % H1_errs[i]] for i in range(len(hh))])

    print(table)

    import matplotlib.pyplot as plt
    plot(u)
    plt.show()

    
    print("fin")
