#%%
from mpi4py import MPI
from dolfinx.fem import Constant
from ufl import SpatialCoordinate, variable, as_vector, grad, curl, div, diff
from utils import L2_norm, create_mesh_fenics
from solver_non_sym import solver
import ufl
from dolfinx import fem

comm = MPI.COMM_WORLD
degree = 1
n = 8

ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 100  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

boundaries = {"bottom": 1, "top": 2, "front": 3, "right": 4, "back": 5, "left": 6}
domain,ft, ct = create_mesh_fenics(comm, n, boundaries)

x = SpatialCoordinate(domain)
t = variable(fem.Constant(domain, ti))
nu = 1.0
sigma = 1.0

def exact(x, t):
    return as_vector((
        x[1]**2 + x[0] * t, 
        x[2]**2 + x[1] * t, 
        x[0]**2 + x[2] * t))

def exact1(x):
    return (x[0]**2) + (x[1]**2) + (x[2]**2)

uex = exact(x,t)
uex1 = exact1(x)

#Manually calculating the RHS

# f0 = ufl.as_vector((
#     -2 + 3*x[0],
#     -2 + 3*x[1],
#     -2 + 3*x[2])
# )

# f1 = fem.Constant(domain, -9.0)

f0 = nu*curl(curl(uex)) + sigma*diff(uex,t) + sigma*grad(uex1)
f1 = -div(sigma*grad(uex1)) - div(sigma*diff(uex,t))

bc_dict = {
    "V": {
        # Interpolate vector-valued exact solution on boundaries
        (boundaries["bottom"], boundaries["top"], 
        boundaries["front"], boundaries["right"],
        boundaries["back"], boundaries["left"]): uex
    },
    "V1": {
        # Interpolate scalar exact solution on all boundaries
        (boundaries["bottom"], boundaries["top"], 
        boundaries["front"], boundaries["right"],
        boundaries["back"], boundaries["left"]): uex1
    }
}

# preconditioner = 'AMS'
preconditioner = 'AMS'

domain_tags = ct


E,B = solver(comm, domain, ft, ct, degree, nu, sigma, t, d_t, num_steps, f0, f1, bc_dict, uex, uex1, preconditioner)

t_prev = T - d_t

uex_prev = exact(x, t_prev)
uex_final = exact(x, T)

da_dt_exact = (uex_final - uex_prev) / d_t
E_exact = -grad(uex1) - da_dt_exact

print("E field error", L2_norm(E - E_exact))
print("B field error", L2_norm(B - curl(uex)))
# %%


# %%
