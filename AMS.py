
from mpi4py import MPI
import numpy
import ufl
from petsc4py import PETSc
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem import functionspace, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, apply_lifting, set_bc
from ufl import SpatialCoordinate, sin, pi, grad, div, variable, diff, dx, cos, curl
from dolfinx.fem import Function, Expression, dirichletbc, form
import numpy as np
from basix.ufl import element
from ufl.core.expr import Expr
from scipy.linalg import norm
from dolfinx.cpp.fem.petsc import (discrete_gradient,
                                   interpolation_matrix)
from dolfinx.io import VTXWriter

iteration_count = []
residual_norm = []

def monitor(ksp, its, rnorm):
    iteration_count.append(its)
    residual_norm.append(rnorm)
    print("Iteration: {}, preconditioned residual: {}".format(its, rnorm))

t = 0  # Start time
T = 0.1  # End time
num_steps = 400  # Number of time steps
d_t = (T - t) / num_steps  # Time step size

n = 8
degree = 1

domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, mesh.CellType.hexahedron)
t = variable(fem.Constant(domain, d_t))
dt = fem.Constant(domain, d_t)

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
V = functionspace(domain, nedelec_elem)

V1 = functionspace(domain, ("Lagrange", 1))

x = SpatialCoordinate(domain)

nu = fem.Constant(domain, default_scalar_type(10))
DG = functionspace(domain, ("DG", 0))
sigma = Function(DG)
sigma.interpolate(lambda x: np.where(x[0] <= 0.5, 5000.0, 1e-12))

def exact(x, t):
    return ufl.as_vector((cos(pi * x[1]) * sin(pi * t), cos(pi * x[2]) * sin(pi * t), cos(pi * x[0]) * sin(pi * t)))

# def exact1(x):
#    return 1 + x[0]**2 + 2*x[1]**2 + 2*x[2]**2

def exact1(x):
    return sin(pi*x[0]) * sin(pi*x[1]) * sin(pi*x[2])

uex = exact(x,t)
uex1 = exact1(x)

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

u_n = fem.Function(V)
uex_expr = Expression(uex, V.element.interpolation_points())
u_n.interpolate(uex_expr)

u_n1 = fem.Function(V1)
uex_expr1 = Expression(uex1, V1.element.interpolation_points())
u_n1.interpolate(uex_expr1)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f0 = nu*curl(curl(uex)) + sigma*diff(uex,t) + sigma*grad(uex1)
a00 = dt*nu*ufl.inner(curl(u), curl(v)) * dx + sigma*ufl.inner(u, v) * dx
L0 = dt* ufl.inner(f0, v) * dx + sigma*ufl.inner(u_n, v) * dx

u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)

f1 = -div(sigma*grad(uex1)) - div(sigma*diff(uex,t))
a11 = ufl.inner(sigma*ufl.grad(u1), ufl.grad(v1)) * ufl.dx
L1 = f1 * v1 * ufl.dx + sigma*ufl.inner(grad(v1), u_n) * ufl.dx

a01 = dt * ufl.inner(sigma*grad(u1), v) * dx
a10 = ufl.inner(sigma*grad(v1), u) * dx

a = form([[a00, a01], [a10, a11]])

A_mat = assemble_matrix_block(a, bcs = bc)
A_mat.assemble()

L = form([L0, L1])

b = assemble_vector_block(L, a, bcs = bc)

a_p = form([[a00, None], [None, a11]])

P = assemble_matrix_block(a_p, bcs = bc)
P.assemble()

A_map = V.dofmap.index_map
V_map = V1.dofmap.index_map

offset_A = A_map.local_range[0] * V.dofmap.index_map_bs + V_map.local_range[0]
offset_S = offset_A + A_map.size_local * V.dofmap.index_map_bs
is_A = PETSc.IS().createStride(A_map.size_local * V.dofmap.index_map_bs, offset_A, 1, comm=PETSc.COMM_SELF)
is_S = PETSc.IS().createStride(V_map.size_local, offset_S, 1, comm=PETSc.COMM_SELF)

#Iterative solvers

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A_mat, P)
ksp.setType("gmres")
ksp.setTolerances(rtol=1e-9)

pc = ksp.getPC()
pc.setType("fieldsplit")
pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
pc.setFieldSplitIS(("A", is_A), ("S", is_S))
ksp_A, ksp_S = ksp.getPC().getFieldSplitSubKSP()

ksp_0, ksp_1 = pc.getFieldSplitSubKSP()
pc0 = ksp_0.getPC()

pc0.setType("hypre")
pc0.setHYPREType("ams")

W = fem.functionspace(domain, ("Lagrange", degree))
G = discrete_gradient(W._cpp_object, V._cpp_object)
G.assemble()
pc0.setHYPREDiscreteGradient(G)

if degree == 1:
    cvec_0 = Function(V)
    cvec_0.interpolate(
        lambda x: np.vstack((np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0])))
    )
    cvec_1 = Function(V)
    cvec_1.interpolate(
        lambda x: np.vstack((np.zeros_like(x[0]), np.ones_like(x[0]), np.zeros_like(x[0])))
    )
    cvec_2 = Function(V)
    cvec_2.interpolate(
        lambda x: np.vstack((np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0])))
    )
    pc0.setHYPRESetEdgeConstantVectors(cvec_0.vector, cvec_1.vector, cvec_2.vector)
else:
    shape = (domain.geometry.dim,)
    Q = fem.functionspace(domain, ("Lagrange", degree, shape))
    Pi = interpolation_matrix(Q._cpp_object, V._cpp_object)
    Pi.assemble()
    pc0.setHYPRESetInterpolations(dim=domain.geometry.dim, ND_Pi_Full=Pi)

pc0.setHYPRESetBetaPoissonMatrix(None)

ksp_1.setType("preonly")
pc1 = ksp_1.getPC()
pc1.setType("gamg")

# ksp.setMonitor(monitor)
ksp.setUp()

ksp_0.setUp()
pc0.setUp()

ksp_1.setUp()
pc1.setUp()

offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs


# Post pro

X = fem.functionspace(domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,)))
A_vis = fem.Function(X)
A_vis.interpolate(u_n)

A_file = VTXWriter(domain.comm, "A.bp", A_vis, "BP4")
A_file.write(t.expression().value)

B_vis = fem.Function(X)
B_3D = curl(u_n)
Bexpr = fem.Expression(B_3D, X.element.interpolation_points())
B_vis.interpolate(Bexpr)

B_file = VTXWriter(domain.comm, "B.bp", B_vis, "BP4")
B_file.write(t.expression().value)

V_file = VTXWriter(domain.comm, "V.bp", u_n1, "BP4")
V_file.write(t.expression().value)

J_vis = fem.Function(X)
J_file = VTXWriter(domain.comm, "J.bp", J_vis, "BP4")
J_file.write(t.expression().value)

for n in range(num_steps):
    t.expression().value += d_t

    u_n_prev = u_n.copy()
    
    u_bc_V.interpolate(u_bc_expr_V)
    u_bc_V1.interpolate(u_bc_expr_V1)

    b = assemble_vector_block(L, a, bcs=bc)

    sol = A_mat.createVecRight()
    ksp.solve(b, sol)

    u_n.x.array[:] = sol.array[:offset]
    u_n1.x.array[:] = sol.array[offset:]

    A_vis.interpolate(u_n)
    A_file.write(t.expression().value)

    B_vis.interpolate(Bexpr)
    B_file.write(t.expression().value)

    V_file.write(t.expression().value)

    E = -(u_n - u_n_prev)/dt - ufl.grad(u_n1)
    J = sigma * E
    
    J_expr = Expression(J, X.element.interpolation_points())
    J_vis.interpolate(J_expr)
    J_file.write(t.expression().value)


    u_n.x.scatter_forward()
    u_n1.x.scatter_forward()

def L2_norm(v: Expr):
    """Computes the L2-norm of v"""
    return np.sqrt(MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(ufl.inner(v, v) * ufl.dx)), op=MPI.SUM))

print(L2_norm(curl(u_n - uex)))
print(L2_norm(u_n1 - uex1))
