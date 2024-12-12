from mpi4py import MPI
from ufl import SpatialCoordinate, variable, as_vector, grad, curl, div, diff, sin, cos, pi, dot, cross, FacetNormal
from solver_non_sym import solver
from dolfinx import fem
from utils import L2_norm, create_mesh_fenics, create_mesh_gmsh

comm = MPI.COMM_WORLD
degree = 1
n = 4

ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 100  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

boundaries = {"bottom": 1, "top": 2, "front": 3, "right": 4, "back": 5, "left": 6}
domain, ft, ct = create_mesh_fenics(comm, n, boundaries)

x = SpatialCoordinate(domain)
t = variable(fem.Constant(domain, ti))
nu_ = 1.0
sigma_ = 1.0

# def exact(x, t):
#     return as_vector(
#         (
#             cos(pi * x[1]) * sin(pi * t),
#             cos(pi * x[2]) * sin(pi * t),
#             cos(pi * x[0]) * sin(pi * t),
#         )
#     )


# def exact1(x):
#     return sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2])

def exact(x, t):
    return as_vector((
        x[1]**2 + x[0] * t,
        x[2]**2 + x[1] * t,
        x[0]**2 + x[2] * t))

def exact1(x):
    return (x[0]**2) + (x[1]**2) + (x[2]**2)

uex = exact(x, t)
uex1 = exact1(x)

f0 = nu_ * curl(curl(uex)) + sigma_ * diff(uex, t) + sigma_ * grad(uex1)
f1 = -div(sigma_ * grad(uex1)) - div(sigma_ * diff(uex, t))

preconditioner = "AMS"

domain_tags = ct
facet_tags = ft

norm = FacetNormal(domain)
hn0 = cross(nu_ * curl(uex), norm)
hn1 = -dot(norm, sigma_ * diff(uex, t) + sigma_ * grad(uex1))

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
    "postpro": True,
    "save_frequency": 10,
    "output_fields": ["A", "B", "V", "J", "E"]
}

E,B,J = solver(comm, domain, ft, ct, degree, nu_, sigma_, t, d_t, num_steps, f0, f1, bc_dict, hn0, hn1, uex, uex1, preconditioner, results)

t_prev = T - d_t

uex_prev = exact(x, t_prev)
uex_final = exact(x, T)

da_dt_exact = (uex_final - uex_prev) / d_t
E_exact = -grad(uex1) - da_dt_exact

print("E field error", L2_norm(E - E_exact))
print("B field error", L2_norm(B - curl(uex)))
