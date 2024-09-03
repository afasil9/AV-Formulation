
#%%
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import ds, dx, inner, grad, div, curl
from dolfinx import fem, io, mesh, plot, default_scalar_type
from dolfinx.fem import dirichletbc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from ufl.core.expr import Expr
from petsc4py import PETSc
from ufl import SpatialCoordinate, as_vector, sin, cos, pi 
from dolfinx.fem import (assemble_scalar, form, Function)
from matplotlib import pyplot as plt
from dolfinx.fem.petsc import assemble_matrix
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
from dolfinx.mesh import locate_entities_boundary
from ufl import (
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    curl,
    grad,
    inner,
    cross,
)
import ufl
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from slepc4py import SLEPc
from dolfinx.mesh import locate_entities_boundary, create_unit_cube, transfer_meshtag, RefinementOption, refine_plaza, GhostMode

iteration_count = []
residual_norm = []

n = 5
degree = 1

def L2_norm(v: Expr):
    """Computes the L2-norm of v"""
    return np.sqrt(MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(inner(v, v) * dx)), op=MPI.SUM))

def monitor(ksp, its, rnorm):
        iteration_count.append(its)
        residual_norm.append(rnorm)
        print("Iteration: {}, preconditioned residual: {}".format(its, rnorm))

domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)

num_steps = 100
te = 1

dt = fem.Constant(domain, te / num_steps)

t = ufl.variable(fem.Constant(domain, 0.1))

gdim = domain.geometry.dim
facet_dim = gdim - 1

# Define function spaces
nedelec_elem = element("N1curl", domain.basix_cell(), degree)
A_space = fem.functionspace(domain, nedelec_elem)

lagrange_element = element("Lagrange", domain.basix_cell(), degree)
V_space = fem.functionspace(domain, lagrange_element)

# Magnetic Vector Potential
A  = ufl.TrialFunction(A_space)
v = ufl.TestFunction(A_space)

# Electric Scalar Potential
S = TrialFunction(V_space)   
q = TestFunction(V_space)

def boundary_marker(x):
    """Marker function for the boundary of a unit cube"""
    # Collect boundaries perpendicular to each coordinate axis
    boundaries = [
        np.logical_or(np.isclose(x[i], 0.0), np.isclose(x[i], 1.0))
        for i in range(3)]
    return np.logical_or(np.logical_or(boundaries[0],
                                        boundaries[1]),
                            boundaries[2])

facets = mesh.locate_entities_boundary(domain, dim=facet_dim,
                                        marker= boundary_marker)
bdofs0 = fem.locate_dofs_topological(A_space, entity_dim=facet_dim, entities=facets)
bdofs1 = fem.locate_dofs_topological(V_space, entity_dim=facet_dim, entities=facets)

# Define Exact Solutions for Magnetic Vector Potential and Electric Scalar Potential

def A_ex(x, t):
    sinx = sin(pi * x[0])
    siny = sin(pi * x[1])
    sinz = sin(pi * x[2])
    sint = sin(pi * t)
    return as_vector((sinx + sint, siny + sint, sinz + sint))

def V_ex(x, t):
    cosx = cos(pi * x[0])
    cosy = cos(pi * x[1])
    cosz = cos(pi * x[2])
    cost = cos(pi * t)
    return cosx+cosy+cosz +cost

x = SpatialCoordinate(domain)
aex = A_ex(x, t)
vex = V_ex(x, t)
# Impose boundary conditions on the exact solution
u_bc_expr_A = Expression(aex, A_space.element.interpolation_points())
u_bc_A = Function(A_space)
u_bc_A.interpolate(u_bc_expr_A)
bc0_ex = dirichletbc(u_bc_A, bdofs0)

u_bc_expr_V = Expression(vex, V_space.element.interpolation_points())
u_bc_V = Function(V_space)
u_bc_V.interpolate(u_bc_expr_V)
bc1_ex = dirichletbc(u_bc_V, bdofs1)

bc = [bc0_ex, bc1_ex]

mu_R = fem.Constant(domain, default_scalar_type(1.0))
sigma = fem.Constant(domain, default_scalar_type(1.0))

# Weak Form

a00 = dt * (1 / mu_R) * inner(curl(A), curl(v)) * dx
a00 += inner((A*sigma), v) * dx 

a01 = inner((grad(S)* sigma), v) * dx
a10 = inner(grad(q), (A*sigma)) * dx

a11 = sigma * inner(grad(S), grad(q)) * dx

a = form([[a00, a01], [a10, a11]])

# a01 = form(a01)
# a01 = assemble_matrix(a01)
# a01.assemble()

# a10 = form(a10)
# a10 = assemble_matrix(a10)
# a10.assemble()

# new = a10.transpose()
# norm = (new - a01).norm()
# print(norm)


A_mat = assemble_matrix_block(a, bcs = bc)
A_mat.assemble()

A_n = Function(A_space)
S_n = Function(V_space)

print(f"A norm = {A_mat.norm()}")

j_e = dt * (1 / mu_R) * curl(curl(aex)) + sigma*ufl.diff(aex,t) + sigma*grad(ufl.diff(vex,t)) #Eddy Current density
time_l0 = (sigma * A_n + sigma * grad(S_n))
f_time_l0 = j_e + time_l0

f_l1 = inner(div(j_e), q)  # Strong Form


time_l1 = inner(grad(q), sigma*A_n) + inner(grad(q), sigma*grad(S_n))
f_time_l1 = f_l1 + time_l1

L0 = inner(f_time_l0,v) * dx
L1 = f_time_l1 * dx              #inner(fem.Constant(domain, 0.0), q) * dx

L = form([L0, L1])

b = assemble_vector_block(L, a, bcs = bc)
print(f"b norm = {b.norm()}")

A_map = A_space.dofmap.index_map
V_map = V_space.dofmap.index_map

offset_A = A_map.local_range[0] * A_space.dofmap.index_map_bs + V_map.local_range[0]
offset_S = offset_A + A_map.size_local * A_space.dofmap.index_map_bs
is_A = PETSc.IS().createStride(A_map.size_local * A_space.dofmap.index_map_bs, offset_A, 1, comm=PETSc.COMM_SELF)
is_S = PETSc.IS().createStride(V_map.size_local, offset_S, 1, comm=PETSc.COMM_SELF)

a_p = form([[a00, None], [None, a11]])

P = assemble_matrix_block(a_p, bcs = bc)
P.assemble()

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
ksp_0.setType("preonly")
pc0 = ksp_0.getPC()

pc0.setType("hypre")
pc0.setHYPREType("ams")

W = fem.functionspace(domain, ("Lagrange", degree))
G = discrete_gradient(W._cpp_object, A_space._cpp_object)
G.assemble()
pc0.setHYPREDiscreteGradient(G)


if degree == 1:
    cvec_0 = Function(A_space)
    cvec_0.interpolate(
        lambda x: np.vstack((np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0])))
    )
    cvec_1 = Function(A_space)
    cvec_1.interpolate(
        lambda x: np.vstack((np.zeros_like(x[0]), np.ones_like(x[0]), np.zeros_like(x[0])))
    )
    cvec_2 = Function(A_space)
    cvec_2.interpolate(
        lambda x: np.vstack((np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0])))
    )
    pc0.setHYPRESetEdgeConstantVectors(cvec_0.vector, cvec_1.vector, cvec_2.vector)
else:
    shape = (domain.geometry.dim,)
    Q = fem.functionspace(domain, ("Lagrange", degree, shape))
    Pi = interpolation_matrix(Q._cpp_object, A_space._cpp_object)
    Pi.assemble()
    pc0.setHYPRESetInterpolations(dim=domain.geometry.dim, ND_Pi_Full=Pi)

ksp_1.setType("preonly")
pc1 = ksp_1.getPC()
pc1.setType("gamg")


ksp.getPC().setUp()
ksp_A.setUp()
ksp_S.setUp()

ksp.view()


offset = A_space.dofmap.index_map.size_local * A_space.dofmap.index_map_bs


aerr = []
serr = []
res = []

for i in range(num_steps):  

    t += dt

    aex = A_ex(x, t)
    vex = V_ex(x, t)

    u_bc_expr_A = Expression(aex, A_space.element.interpolation_points())
    u_bc_A.interpolate(u_bc_expr_A)
    u_bc_expr_V = Expression(vex, V_space.element.interpolation_points())
    u_bc_V.interpolate(u_bc_expr_V)
    
    # with b.localForm() as loc_b:
    #     loc_b.set(0)

    b = assemble_vector_block(L, a, bcs = bc)
    sol = A_mat.createVecRight()  #Solution Vector    
    ksp.solve(b, sol)
   
    # reason = ksp.getConvergedReason()
    # print(f"Convergence reason {reason}") 

    # print("b norm is ", b.norm())
    # print("sol norm is ", sol.norm())
    
    residual = A_mat * sol - b
    # print('residual is ', residual.norm())
    res.append(residual.norm())

    A_n.x.array[:offset] = sol.array_r[:offset]
    S_n.x.array[:(len(sol.array_r) - offset)] = sol.array_r[offset:]

    a_error = A_n - aex
    s_error = S_n - vex

    # print(sol.norm())
    # aerr.append(L2_norm(a_error))
    # serr.append(L2_norm(s_error))

    print("L2 A is",L2_norm(a_error))
    # print("L2 S is",L2_norm(s_error))

    # print("Exact A norm:", L2_norm(aex))
    # print("Computed A norm:", L2_norm(A_n))
    # print("Exact V norm:", L2_norm(vex))
    # print("Computed V norm:", L2_norm(S_n))


# plt.figure()
# plt.plot(aerr)
# plt.title("A error")

# plt.figure()
# plt.plot(serr)
# plt.title("S error")

# plt.figure()
# plt.plot(res)
# plt.title("Residual")  

# plt.show()

# V0 = fem.functionspace(domain, ("Discontinuous Lagrange", 2, (gdim,)))
# A_result = Function(V0, dtype=np.float64)
# A_result.interpolate(A_n)
# with VTXWriter(domain.comm, "output_A.bp", A_result , "bp4") as f:
#     f.write(0.0)

# with VTXWriter(domain.comm, "output_S.bp", S_n , "bp4") as f:
#     f.write(0.0)
# %%

# %%


