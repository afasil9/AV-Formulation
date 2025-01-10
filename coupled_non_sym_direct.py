import numpy as np
from basix.ufl import element
from dolfinx import default_scalar_type, mesh
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    curl,
    diff,
    div,
    dx,
    grad,
    inner,
    variable,
)

from utils import L2_norm

comm = MPI.COMM_WORLD
degree = 1
n = 4

ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 100  # Number of time steps
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

x = SpatialCoordinate(domain)


def exact(x, t):
    return as_vector((x[1] ** 2 + x[0] * t, x[2] ** 2 + x[1] * t, x[0] ** 2 + x[2] * t))


def exact1(x):
    return (x[0] ** 2) + (x[1] ** 2) + (x[2] ** 2)


x = SpatialCoordinate(domain)

uex = exact(x, t)
uex1 = exact1(x)

# Manually calculating the RHS
# f0 = as_vector((-2 + 3 * x[0], -2 + 3 * x[1], -2 + 3 * x[2]))
# f1 = Constant(domain, -9.0)

f0 = nu * curl(curl(uex)) + sigma * diff(uex, t) + sigma * grad(uex1)
f1 = -div(sigma * grad(uex1)) - div(sigma * diff(uex, t))

u_n = Function(V)
u_expr = Expression(uex, V.element.interpolation_points())
u_n.interpolate(u_expr)

u_n1 = Function(V1)
uex_expr1 = Expression(uex1, V1.element.interpolation_points())
u_n1.interpolate(uex_expr1)


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
u_bc_expr_V1 = Expression(uex1, V1.element.interpolation_points())
u_bc_V1 = Function(V1)
u_bc_V1.interpolate(u_bc_expr_V1)
bc_ex1 = dirichletbc(u_bc_V1, bdofs1)

bc = [bc_ex, bc_ex1]

u = TrialFunction(V)
v = TestFunction(V)

u1 = TrialFunction(V1)
v1 = TestFunction(V1)

a00 = dt * nu * inner(curl(u), curl(v)) * dx + sigma * inner(u, v) * dx
L0 = dt * inner(f0, v) * dx + sigma * inner(u_n, v) * dx

a01 = dt * sigma * inner(grad(u1), v) * dx
a10 = sigma * inner(grad(v1), u) * dx

a11 = dt * inner(sigma * grad(u1), grad(v1)) * dx
L1 = dt * f1 * v1 * dx + sigma * inner(grad(v1), u_n) * dx

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
opts["mat_mumps_icntl_24"] = 1
opts["mat_mumps_icntl_25"] = 0
opts["ksp_error_if_not_converged"] = 1
ksp.setFromOptions()

uh, uh1 = Function(V), Function(V1)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

sol = A_mat.createVecRight()
ksp.solve(b, sol)

uh.x.array[:] = sol.array_r[:offset]
uh1.x.array[:] = sol.array_r[offset:]

u_n.x.array[:] = uh.x.array
u_n1.x.array[:] = uh1.x.array

t_prev = variable(Constant(domain, ti - d_t))

for n in range(num_steps):

    t.expression().value += d_t
    t_prev.expression().value += d_t

    u_n_prev = u_n.copy()
    u_n1_prev = u_n1.copy()

    uex_prev = exact(x, t_prev)

    u_bc_V.interpolate(u_bc_expr_V)
    u_bc_V1.interpolate(u_bc_expr_V1)

    b = assemble_vector_block(L, a, bcs=bc)

    sol = A_mat.createVecRight()
    ksp.solve(b, sol)

    uh.x.array[:] = sol.array_r[:offset]
    uh1.x.array[:] = sol.array_r[offset:]

    u_n.x.array[:] = uh.x.array
    u_n1.x.array[:] = uh1.x.array

    u_n.x.scatter_forward()
    u_n1.x.scatter_forward()

    uex = exact(x, t)

da_dt_exact = (uex - uex_prev) / dt
E_exact = -grad(uex1) - da_dt_exact

da_dt = (u_n - u_n_prev) / dt
E = -grad(u_n1) - da_dt

print("E field error", L2_norm(E - E_exact))
print("B field error", L2_norm(curl(u_n - uex)))
