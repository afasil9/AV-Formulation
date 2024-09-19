from mpi4py import MPI
import numpy
import ufl
from petsc4py import PETSc
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem import functionspace, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, apply_lifting, set_bc
from ufl import SpatialCoordinate, sin, pi, grad, div, variable, diff, dx, cos, as_vector,curl, inner
from dolfinx.fem import Function, Expression, dirichletbc, form
import numpy as np
from ufl.core.expr import Expr
from basix.ufl import element

t = 0  # Start time
T = 0.1  # End time
num_steps = 100  # Number of time steps
d_t = (T - t) / num_steps  # Time step size

n = 8
degree = 1

domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, mesh.CellType.hexahedron)
t = variable(fem.Constant(domain, d_t))
dt = fem.Constant(domain, d_t)

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
V = functionspace(domain, nedelec_elem)
lagrange_element = element("Lagrange", domain.basix_cell(), degree)
V1 = functionspace(domain, lagrange_element)

x = SpatialCoordinate(domain)

def exact(x, t):
    return as_vector((cos(pi * x[1]) * sin(pi * t), cos(pi * x[2]) * sin(pi * t), cos(pi * x[0]) * sin(pi * t)))

def exact1(x, t):
    return sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2]) * sin(pi * t)

uex = exact(x,t)
uex1 = exact1(x,t)

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

u_n = Function(V)
uex_expr = Expression(uex, V.element.interpolation_points())
u_n.interpolate(uex_expr)

u_n1 = Function(V1)
uex_expr1 = Expression(uex1, V1.element.interpolation_points())
u_n1.interpolate(uex_expr1)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)

f = curl(curl(uex)) + diff(uex, t) + grad(diff(uex1,t))
a00 = dt * inner(curl(u), curl(v)) * dx + inner(u, v) * dx 
L0 = inner(f, v) * dt * dx + inner(u_n, v) * dx + inner(grad(u_n1), v) * dx

a01 = inner(grad(u1), v) * dx 
a10 = inner(grad(v1), u) * dx

f1 = -div(f)
a11 = inner(grad(u1), grad(v1)) * dt * dx + inner(u1, v1) * dx
L1 = inner(f1, v1) * dt * dx + inner(u_n1, v1) * dx

a = form([[a00, a01], [a10, a11]])

A_mat = assemble_matrix_block(a, bcs = bc)
A_mat.assemble()

L = form([L0, L1])

b = assemble_vector_block(L, a, bcs = bc)

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A_mat)
ksp.setType("preonly")

pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")

opts = PETSc.Options()  # type: ignore
opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
opts["ksp_error_if_not_converged"] = 1
ksp.setFromOptions()

uh, uh1 = Function(V), Function(V1)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

sol = A_mat.createVecRight()

ksp.solve(b, sol)

for n in range(num_steps):
    t.expression().value += d_t
    
    u_bc_V.interpolate(u_bc_expr_V)
    u_bc_V1.interpolate(u_bc_expr_V1)

    b = assemble_vector_block(L, a, bcs=bc)

    sol = A_mat.createVecRight()
    ksp.solve(b, sol)
    
    u_n.x.array[:] = sol.array[:offset]
    u_n1.x.array[:] = sol.array[offset:]

    u_n.x.scatter_forward()
    u_n1.x.scatter_forward()


def L2_norm(v: Expr):
    """Computes the L2-norm of v"""
    return np.sqrt(MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(inner(v, v) * ufl.dx)), op=MPI.SUM))

print('Vector potential error', L2_norm(curl(u_n) - curl(uex)))
print('Scalar potential error', L2_norm(u_n1 - uex1))
