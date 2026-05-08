from dolfin import *

import sympy as sp

from prettytable import PrettyTable
import matplotlib.pyplot as plt


# ----- GLOBAL VARIABLES -----

# Constants
a           = 0
b           = 1
epsilon     = 1e-3
alpha       = 1
beta        = 0
u0          = 0
u1          = 0


# Symbolic functions
x = sp.symbols("x[0]")

f = 1

p = (u1-u0)/(b-a)*(x-a) + u0
u_ex_sp = x - (sp.exp(-(1-x)/epsilon) - sp.exp(-1/epsilon)) / (1 - sp.exp(-1/epsilon))


# Turn symbolic functions to FEniCS expressions
f = Expression(sp.printing.ccode(f), degree = 5)
u_ex = Expression(sp.printing.ccode(u_ex_sp), degree = 5)


# ----- HELPER FUNCTIONS -----


def boundary(x):
    return x[0] < a + DOLFIN_EPS or x[0] > b - DOLFIN_EPS

def plot_sol(expr, n):
    mesh = IntervalMesh(n, a, b)
    V = FunctionSpace(mesh, "Lagrange", 3)

    pr = project(expr, V)
    plot(pr, ls="--", label="Solucion")

def save_table(file_name, hh, L2, H1):
    file = open(f"{file_name}", "w")

    for i in range(len(hh)):
        file.write(f"{hh[i]} {L2[i]} {H1[i]}\n")

    file.close()


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
    L2_err = errornorm(u, project(u_ex, V), norm_type="L2")
    H1_err = errornorm(u, project(u_ex, V), norm_type="H1")

    return u, mesh.hmax(), L2_err, H1_err


# ----- MAIN PROGRAM -----


if __name__ == "__main__":
    degree  = 2
    NN      = [8, 16, 32, 64, 128, 256, 512]

    u       = None
    hh      = list()
    L2_errs = list()
    H1_errs = list()


    # Iterations and plotting

    for n in NN:
        u, h, L2_err, H1_err = vf_solve(n, degree)

        hh.append(h)
        L2_errs.append(L2_err)
        H1_errs.append(H1_err)
        plot(u, linewidth=2, label=f"h = {h:.3f}")

    plot_sol(u_ex, 256) 

    plt.title(f"Grado: {degree} - ε: {epsilon}")
    plt.legend()

    # Saving image (optional)
    # plt.savefig(f"Plots/deg {degree} - eps {epsilon}.png")

    plt.show()


    # Tables

    table           = PrettyTable(["h", "err_L2", "err_H1"])
    table.border    = True

    table.add_rows([["%.3f" % hh[i], "%2.2e" % L2_errs[i], "%2.2e" % H1_errs[i]] for i in range(len(hh))])

    print(table)


    # Tables for rates

    # Saving table (optional)
    # save_table(f"Tables/deg {degree} - eps {epsilon}.txt", hh, L2_errs, H1_errs)

    print("Listo!")

