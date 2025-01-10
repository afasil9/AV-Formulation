import numpy as np
from basix.ufl import element
from dolfinx import default_scalar_type, mesh
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
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

a_p = form([[a00, None], [None, a11]])
P = assemble_matrix_block(a_p, bcs=bc)
P.assemble()

u_map = V.dofmap.index_map
u1_map = V1.dofmap.index_map

offset_u = u_map.local_range[0] * V.dofmap.index_map_bs + u1_map.local_range[0]
offset_u1 = offset_u + u_map.size_local * V.dofmap.index_map_bs

is_u = PETSc.IS().createStride(
    u_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=PETSc.COMM_SELF
)
is_u1 = PETSc.IS().createStride(u1_map.size_local, offset_u1, 1, comm=PETSc.COMM_SELF)


ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A_mat, P)
ksp.setType("gmres")
ksp.setTolerances(rtol=1e-10)
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
ksp.getPC().setFieldSplitIS(("u", is_u), ("u1", is_u1))
ksp_u, ksp_u1 = ksp.getPC().getFieldSplitSubKSP()

# Preconditioner for u

ksp_u.setType("preonly")
pc0 = ksp_u.getPC()
pc0.setType("hypre")
pc0.setHYPREType("ams")

V_CG = functionspace(domain, ("CG", degree))._cpp_object
G = discrete_gradient(V_CG, V._cpp_object)
G.assemble()
pc0.setHYPREDiscreteGradient(G)

if degree == 1:
    cvec_0 = Function(V)
    cvec_0.interpolate(
        lambda x: np.vstack(
            (np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_1 = Function(V)
    cvec_1.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.ones_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_2 = Function(V)
    cvec_2.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0]))
        )
    )
    pc0.setHYPRESetEdgeConstantVectors(cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec)
else:
    Vec_CG = functionspace(domain, ("CG", degree, (domain.geometry.dim,)))
    Pi = interpolation_matrix(Vec_CG._cpp_object, V._cpp_object)
    Pi.assemble()

    # Attach discrete gradient to preconditioner
    pc0.setHYPRESetInterpolations(domain.geometry.dim, None, None, Pi, None)

opts = PETSc.Options()
opts[f"{ksp_u.prefix}pc_hypre_ams_cycle_type"] = 7
opts[f"{ksp_u.prefix}pc_hypre_ams_tol"] = 0
opts[f"{ksp_u.prefix}pc_hypre_ams_max_iter"] = 1
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_beta_theta"] = 0.25
opts[f"{ksp_u.prefix}pc_hypre_ams_print_level"] = 1
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_alpha_options"] = "10,1,3"
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_beta_options"] = "10,1,3"
opts[f"{ksp_u.prefix}pc_hypre_ams_print_level"] = 0

ksp_u.setFromOptions()

# Preconditioner for u1
ksp_u1.setType("preonly")
pc1 = ksp_u1.getPC()
pc1.setType("gamg")

ksp.setUp()
pc0.setUp()
pc1.setUp()

uh, uh1 = Function(V), Function(V1)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

sol = A_mat.createVecRight()
ksp.solve(b, sol)

uh.x.array[:] = sol.array_r[:offset]
uh1.x.array[:] = sol.array_r[offset:]

u_n.x.array[:] = uh.x.array
w_n.x.array[:] = uh1.x.array

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
