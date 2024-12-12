from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from dolfinx import default_scalar_type, mesh
from dolfinx.fem.petsc import assemble_vector_block, assemble_matrix_block
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from basix.ufl import element
from dolfinx.fem import (
    dirichletbc,
    form,
    Function,
    Expression,
    locate_dofs_topological,
    functionspace,
    Constant,
)
from ufl import (
    TrialFunction,
    TestFunction,
    inner,
    grad,
    div,
    curl,
    variable,
    as_vector,
    diff,
    sin,
    cos,
    pi,
    SpatialCoordinate,
    dx,
)
from utils import L2_norm


comm = MPI.COMM_WORLD
degree = 1

n = 4

ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 200  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
gdim = domain.geometry.dim
facet_dim = gdim - 1  # Topological dimension

t = variable(Constant(domain, ti))
dt = Constant(domain, d_t)

nu = Constant(domain, default_scalar_type(1.0))
sigma = Constant(domain, default_scalar_type(1.0))

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
V = functionspace(domain, nedelec_elem)
lagrange_elem = element("Lagrange", domain.basix_cell(), degree)
V1 = functionspace(domain, lagrange_elem)

def exact(x, t):
    return as_vector((x[1] ** 2 + x[0] * t, x[2] ** 2 + x[1] * t, x[0] ** 2 + x[2] * t))

def exact1(x, t):
    return (x[0] ** 2) * t + (x[1] ** 2) * t + (x[2] ** 2) * t

x = SpatialCoordinate(domain)

uex = exact(x, t)
wex = exact1(x, t)

f0 = nu * curl(curl(uex)) + sigma * diff(uex, t) + sigma * grad(diff(wex, t))
f1 = -div(f0)

u_n = Function(V)
u_expr = Expression(uex, V.element.interpolation_points())
u_n.interpolate(u_expr)

w_n = Function(V1)
wex_expr = Expression(wex, V1.element.interpolation_points())
w_n.interpolate(wex_expr)


def boundary_marker(x):
    """Marker function for the boundary of a unit cube"""
    # Collect boundaries perpendicular to each coordinate axis
    boundaries = [
        np.logical_or(np.isclose(x[i], 0.0), np.isclose(x[i], 1.0)) for i in range(3)
    ]
    return np.logical_or(np.logical_or(boundaries[0], boundaries[1]), boundaries[2])


gdim = domain.geometry.dim
facet_dim = gdim - 1

facets = mesh.locate_entities_boundary(domain, dim=facet_dim, marker=boundary_marker)

bdofs0 = locate_dofs_topological(V, entity_dim=facet_dim, entities=facets)
u_bc_expr_V = Expression(uex, V.element.interpolation_points())
u_bc_V = Function(V)
u_bc_V.interpolate(u_bc_expr_V)
bc_ex = dirichletbc(u_bc_V, bdofs0)

bdofs1 = locate_dofs_topological(V1, entity_dim=facet_dim, entities=facets)
w_bc_expr_V1 = Expression(wex, V1.element.interpolation_points())
w_bc_V1 = Function(V1)
w_bc_V1.interpolate(w_bc_expr_V1)
bc_ex1 = dirichletbc(w_bc_V1, bdofs1)

bc = [bc_ex, bc_ex1]

u = TrialFunction(V)
v = TestFunction(V)

w1 = TrialFunction(V1)
v1 = TestFunction(V1)

a00 = dt * nu * inner(curl(u), curl(v)) * dx + sigma * inner(u, v) * dx
L0 = (
    dt * inner(f0, v) * dx
    + sigma * inner(u_n, v) * dx
    + sigma * inner(v, grad(w_n)) * dx
)

a01 = sigma * inner(grad(w1), v) * dx
a10 = sigma * inner(grad(v1), u) * dx

a11 = dt * inner(sigma * grad(w1), grad(v1)) * dx
L1 = (
    dt * f1 * v1 * dx
    + sigma * inner(grad(v1), u_n) * dx
    + sigma * inner(grad(v1), grad(w_n)) * dx
)

a = form([[a00, a01], [a10, a11]])


A_mat = assemble_matrix_block(a, bcs=bc)
A_mat.assemble()

L = form([L0, L1])
b = assemble_vector_block(L, a, bcs=bc)

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A_mat)
ksp.setType("preonly")

pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")

opts = PETSc.Options()
opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
opts[
    "mat_mumps_icntl_24"
] = 1
opts[
    "mat_mumps_icntl_25"
] = 0  
opts["ksp_error_if_not_converged"] = 1
ksp.setFromOptions()

print("norm of bc", L2_norm(u_bc_V))
print("norm of uex", L2_norm(uex))
print("norm of u_n", L2_norm(u_n))

uh, uh1 = Function(V), Function(V1)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

sol = A_mat.createVecRight()
ksp.solve(b, sol)

uh.x.array[:] = sol.array_r[:offset]
uh1.x.array[:] = sol.array_r[offset:]

u_n.x.array[:] = uh.x.array
w_n.x.array[:] = uh1.x.array

print("norm of u_n after solve", L2_norm(u_n))
print(ksp.getTolerances())

t_prev = variable(Constant(domain, ti - d_t))

for n in range(num_steps):

    t.expression().value += d_t
    t_prev.expression().value += d_t

    u_n_prev = u_n.copy()
    w_n1_prev = w_n.copy()

    uex_prev = exact(x, t_prev)
    wex_prev = exact1(x, t_prev)

    u_bc_V.interpolate(u_bc_expr_V)
    w_bc_V1.interpolate(w_bc_expr_V1)

    b = assemble_vector_block(L, a, bcs=bc)

    sol = A_mat.createVecRight()
    ksp.solve(b, sol)

    uh.x.array[:] = sol.array_r[:offset]
    uh1.x.array[:] = sol.array_r[offset:]

    u_n.x.array[:] = uh.x.array
    w_n.x.array[:] = uh1.x.array

    u_n.x.scatter_forward()
    w_n.x.scatter_forward()

    uex = exact(x, t)
    wex = exact1(x, t)

da_dt_exact = (uex - uex_prev) / dt
dw_dt_exact = (wex - wex_prev) / dt
E_exact = -grad(dw_dt_exact) - da_dt_exact

da_dt = (u_n - u_n_prev) / dt
dw_dt = (w_n - w_n1_prev) / dt
E = -grad(dw_dt) - da_dt

print("E field error", L2_norm(E - E_exact))
print("B field error", L2_norm(curl(u_n - uex)))