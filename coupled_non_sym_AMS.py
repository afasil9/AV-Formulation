#%%
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import ds, dx, grad, inner, curl
from dolfinx import fem, io, mesh, plot, default_scalar_type
from dolfinx.fem import dirichletbc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from ufl.core.expr import Expr
from petsc4py import PETSc
from ufl import SpatialCoordinate, as_vector, sin, pi, curl
from dolfinx.fem import (assemble_scalar, form, Function)
from matplotlib import pyplot as plt
from dolfinx.fem.petsc import assemble_matrix, assemble_vector_block, assemble_matrix_block
from dolfinx.fem import petsc, Expression, locate_dofs_topological
from dolfinx.io import VTXWriter
from dolfinx.cpp.fem.petsc import (discrete_gradient,
                                   interpolation_matrix)
from basix.ufl import element
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
from basix.ufl import element
from dolfinx import fem, io, la, default_scalar_type
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.fem import Function, form, locate_dofs_topological, petsc
from ufl import (
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    curl,
    cos,
    inner,
    cross,
)
import ufl
import sys
from ufl import variable
from scipy.linalg import norm
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
facet_dim = gdim - 1 #Topological dimension 

t = variable(fem.Constant(domain, ti))
dt = fem.Constant(domain, d_t)

nu = fem.Constant(domain, default_scalar_type(1.0))
sigma = fem.Constant(domain, default_scalar_type(1.0))

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
V = fem.functionspace(domain, nedelec_elem)
lagrange_elem = element("Lagrange", domain.basix_cell(), degree)
V1 = fem.functionspace(domain, lagrange_elem)

x = SpatialCoordinate(domain)

def exact(x, t):
    return as_vector((
        x[1]**2 + x[0] * t, 
        x[2]**2 + x[1] * t, 
        x[0]**2 + x[2] * t))


def exact1(x):
    return (x[0]**2) + (x[1]**2) + (x[2]**2)

x = SpatialCoordinate(domain)

uex = exact(x,t)
uex1 = exact1(x)

#Manually calculating the RHS

f0 = ufl.as_vector((
    -2 + 3*x[0],
    -2 + 3*x[1],
    -2 + 3*x[2])
)

f1 = fem.Constant(domain, -9.0)

u_n = Function(V)
u_expr = Expression(uex, V.element.interpolation_points())
u_n.interpolate(u_expr)

u_n1 = fem.Function(V1)
uex_expr1 = Expression(uex1, V1.element.interpolation_points())
u_n1.interpolate(uex_expr1)

def boundary_marker(x):
    """Marker function for the boundary of a unit cube"""
    # Collect boundaries perpendicular to each coordinate axis
    boundaries = [
        np.logical_or(np.isclose(x[i], 0.0), np.isclose(x[i], 1.0))
        for i in range(3)]
    return np.logical_or(np.logical_or(boundaries[0],
                                        boundaries[1]),
                            boundaries[2])


gdim = domain.geometry.dim
facet_dim = gdim - 1

facets = mesh.locate_entities_boundary(domain, dim=facet_dim,
                                        marker= boundary_marker)

bdofs0 = fem.locate_dofs_topological(V, entity_dim=facet_dim, entities=facets)
u_bc_expr_V = Expression(uex, V.element.interpolation_points())
u_bc_V = Function(V)
u_bc_V.interpolate(u_bc_expr_V)
bc_ex = dirichletbc(u_bc_V, bdofs0)

bdofs1 = fem.locate_dofs_topological(V1, entity_dim=facet_dim, entities=facets)
u_bc_expr_V1 = Expression(uex1, V1.element.interpolation_points())
u_bc_V1 = Function(V1)
u_bc_V1.interpolate(u_bc_expr_V1)
bc_ex1 = dirichletbc(u_bc_V1, bdofs1)

bc = [bc_ex, bc_ex1]

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)

a00 = dt*nu*ufl.inner(curl(u), curl(v)) * dx + sigma*ufl.inner(u, v) * dx
L0 = dt* ufl.inner(f0, v) * dx + sigma*ufl.inner(u_n, v) * dx 

a01 = dt * sigma*ufl.inner(grad(u1), v) * dx
a10 = sigma*ufl.inner(grad(v1), u) *dx

a11 = dt * ufl.inner(sigma*ufl.grad(u1), ufl.grad(v1)) * ufl.dx
L1 = dt * f1 * v1 * ufl.dx + sigma*ufl.inner(grad(v1),u_n) *dx

a = form([[a00, a01], [a10, a11]])

A_mat = assemble_matrix_block(a, bcs = bc)
A_mat.assemble()

L = form([L0, L1])
b = assemble_vector_block(L, a, bcs = bc)

a_p = form([[a00, None], [None, a11]])
P = assemble_matrix_block(a_p, bcs = bc)
P.assemble()

u_map = V.dofmap.index_map
u1_map = V1.dofmap.index_map

offset_u = u_map.local_range[0] * V.dofmap.index_map_bs + u1_map.local_range[0]
offset_u1 = offset_u + u_map.size_local * V.dofmap.index_map_bs

is_u = PETSc.IS().createStride(u_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=PETSc.COMM_SELF)
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

V_CG = fem.functionspace(domain, ("CG", degree))._cpp_object
G = discrete_gradient(V_CG, V._cpp_object)
G.assemble()
pc0.setHYPREDiscreteGradient(G)

if degree == 1:
    cvec_0 = Function(V)
    cvec_0.interpolate(lambda x: np.vstack((np.ones_like(x[0]),
                                            np.zeros_like(x[0]),
                                            np.zeros_like(x[0]))))
    cvec_1 = Function(V)
    cvec_1.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
                                            np.ones_like(x[0]),
                                            np.zeros_like(x[0]))))
    cvec_2 = Function(V)
    cvec_2.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
                                            np.zeros_like(x[0]),
                                            np.ones_like(x[0]))))
    pc0.setHYPRESetEdgeConstantVectors(cvec_0.vector,
                                        cvec_1.vector,
                                        cvec_2.vector)
else:
    Vec_CG = fem.functionspace(domain, ("CG", degree, (domain.geometry.dim,)))
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
u_n1.x.array[:] = uh1.x.array

print("norm of u_n after solve", L2_norm(u_n))
print(ksp.getTolerances())


t_prev = variable(fem.Constant(domain, ti-d_t))
#%%

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

    # print("current",L2_norm(u_n1))
    # print("prev",L2_norm(u_n1_prev))

    uex = exact(x, t)

    # print("uex", L2_norm(uex))
    # print("uex prev", L2_norm(uex_prev))


# print("final uex", L2_norm(uex))
# print("final uex prev", L2_norm(uex_prev))

da_dt_exact = (uex - uex_prev) / dt
E_exact = -grad(uex1) - da_dt_exact

da_dt = (u_n - u_n_prev) / dt
E = -grad(u_n1) - da_dt

print("E field error", L2_norm(E - E_exact))
print("B field error", L2_norm(curl(u_n - uex)))