from dolfinx import fem
from mpi4py import MPI
from ufl import (
    FacetNormal,
    SpatialCoordinate,
    as_vector,
    cross,
    curl,
    diff,
    div,
    dot,
    grad,
    variable,
    sin,
    cos,
    pi,
)

from solver_non_sym import solver
from solver_sym import solver_sym
from utils import L2_norm, create_mesh_fenics

# solver_type = "non symmetric"
solver_type = "symmetric"
comm = MPI.COMM_WORLD
degree = 1
n = 4

ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 100  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

boundaries = {"bottom": 1, "top": 2, "front": 3, "right": 4, "back": 5, "left": 6}
domain, ft, ct = create_mesh_fenics(comm, n, boundaries)

# domain, ft, ct = create_mesh_gmsh(comm, n, boundaries)

x = SpatialCoordinate(domain)
t = variable(fem.Constant(domain, ti))
nu_ = 1.0
sigma_ = 1.0

# Polynomial exact solution

# def exact(x, t):
#     return as_vector((
#         x[1]**2 + x[0] * t,
#         x[2]**2 + x[1] * t,
#         x[0]**2 + x[2] * t))

# def exact1_non(x):
#     return (x[0]**2) + (x[1]**2) + (x[2]**2)

# def exact1_time(x,t):
#     return (x[0]**2) * t + (x[1]**2) * t + (x[2]**2) * t


def exact(x, t):
    return as_vector(
        (
            cos(pi * x[1]) * sin(pi * t),
            cos(pi * x[2]) * sin(pi * t),
            cos(pi * x[0]) * sin(pi * t),
        )
    )

def exact1_non(x):
    return sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2])


def exact1_time(x, t):
    return sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2]) * sin(pi * t)


uex = exact(x, t)
norm = FacetNormal(domain)
hn0 = cross(nu_ * curl(uex), norm)


if solver_type == "non symmetric":
    uex1 = exact1_non(x)
    f0 = nu_ * curl(curl(uex)) + sigma_ * diff(uex, t) + sigma_ * grad(uex1)
    f1 = -div(f0)
    hn1 = -dot(norm, sigma_ * diff(uex, t) + sigma_ * grad(uex1))
else:
    uex1 = exact1_time(x,t)
    f0 = nu_ * curl(curl(uex)) + sigma_ * diff(uex, t) + sigma_ * grad(diff(uex1, t))
    f1 = -div(f0)
    hn1 = -dot(norm, sigma_ * diff(uex, t) + sigma_ * grad(diff(uex1, t)))


preconditioner = "AMS"

domain_tags = ct
facet_tags = ft


bc_dict = {
    "dirichlet": {
        "V": {
            (
                boundaries["bottom"],
                boundaries["top"],
                boundaries["back"],
                boundaries["left"],
            ): uex
        },
        "V1": {
            (
                boundaries["bottom"],
                boundaries["top"],
                boundaries["back"],
                boundaries["left"],
            ): uex1
        },
    },
    "neumann": {
        "V": {(boundaries["front"], boundaries["right"]): uex},
        "V1": {(boundaries["front"], boundaries["right"]): uex1},
    },
}

results = {
    "postpro": False,
    "save_frequency": 10,
    "output_fields": ["A", "B", "V", "J", "E"]
}

if solver_type == "non symmetric":
    E,B,J = solver(comm, domain, ft, ct, degree, nu_, sigma_, t, d_t, num_steps, f0, f1, bc_dict, hn0, hn1, uex, uex1, preconditioner, results)
    t_prev = T - d_t

    uex_prev = exact(x, t_prev)
    uex_final = exact(x, T)

    da_dt_exact = (uex_final - uex_prev) / d_t
    E_exact = -grad(uex1) - da_dt_exact

    print("E field error", L2_norm(E - E_exact))
    print("B field error", L2_norm(B - curl(uex)))
else:
    E,B,J = solver_sym(comm, domain, ft, ct, degree, nu_, sigma_, t, d_t, num_steps, f0, f1, bc_dict, hn0, hn1, uex, uex1, preconditioner, results)
    t_prev = T - d_t

    uex_prev = exact(x, t_prev)
    uex_final = exact(x, T)

    da_dt_exact = (uex_final - uex_prev) / d_t

    wex_prev = exact1_time(x, t_prev)
    wex_final = exact1_time(x, T)

    dw_dt_exact = (wex_final - wex_prev) / d_t
    E_exact = -grad(dw_dt_exact) - da_dt_exact

    print("E field error", L2_norm(E - E_exact))
    print("B field error", L2_norm(B - curl(uex)))
