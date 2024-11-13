#%%
from petsc4py import PETSc
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
import numpy
from ufl.core.expr import Expr
from dolfinx.fem import assemble_scalar, form, Function, Expression, dirichletbc, functionspace
from ufl import SpatialCoordinate, sin, pi, variable, div, grad, cos, inner
from basix.ufl import element
import numpy as np
from scipy.linalg import norm
from dolfinx import default_scalar_type

def L2_norm(v: Expr):
    """Computes the L2-norm of v"""
    return np.sqrt(MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(ufl.inner(v, v) * ufl.dx)), op=MPI.SUM))

t = 0  # Start time
T = 0.1  # End time
num_steps = 200  # Number of time steps
d_t = (T - t) / num_steps  # Time step size

n = 4
domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)

dt = fem.Constant(domain, d_t)

degree = 1
lagrange_element = element("Lagrange", domain.basix_cell(), degree)
V = functionspace(domain, lagrange_element)

x = SpatialCoordinate(domain)
t = variable(fem.Constant(domain, 0.0))

w_ex_t = (x[0]**2) * t + (x[1]**2) * t + (x[2]**2) * t
f = fem.Constant(domain, -6.0)

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

bdofs1 = fem.locate_dofs_topological(V, entity_dim=facet_dim, entities=facets)

u_bc_expr_V = Expression(w_ex_t, V.element.interpolation_points())
u_bc_V = Function(V)
u_bc_V.interpolate(u_bc_expr_V)
bc1_ex = dirichletbc(u_bc_V, bdofs1)

w_n = fem.Function(V)
uex_expr = Expression(w_ex_t, V.element.interpolation_points())
w_n.interpolate(uex_expr)

S = ufl.TrialFunction(V)
q = ufl.TestFunction(V)

lhs = ufl.inner(grad(S), grad(q)) * ufl.dx
rhs = ufl.inner(f,q) * dt * ufl.dx + ufl.inner(grad(w_n), grad(q)) * ufl.dx

a = fem.form(lhs)
L = fem.form(rhs)

A = assemble_matrix(a, bcs=[bc1_ex])
A.assemble()

b = assemble_vector(L)
apply_lifting(b, [a], [[bc1_ex]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc1_ex])

print(b.norm())

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.getPC().setType(PETSc.PC.Type.LU)

uh = fem.Function(V)

ksp.solve(b, uh.vector)

w_n.x.array[:] = uh.x.array 

# V_file = io.VTXWriter(domain.comm, "scalar.bp", w_n, "BP4")
# V_file.write(t.expression().value)

for n in range(num_steps):
    t.expression().value += dt.value

    # print("time is ", t.expression().value)

    w_n_prev = w_n.copy()

    w_ex_t = (x[0]**2) * t + (x[1]**2) * t + (x[2]**2) * t

    u_bc_expr_V = Expression(w_ex_t, V.element.interpolation_points())
    u_bc_V = Function(V)
    u_bc_V.interpolate(u_bc_expr_V)
    bc1_ex = dirichletbc(u_bc_V, bdofs1)

    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, L)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [a], [[bc1_ex]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc1_ex])

    # print("Exact", L2_norm(s_ex_t))

    # print("Difference", L2_norm(s_n - s_ex_t))

    # Solve linear problem
    ksp.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    w_n.x.array[:] = uh.x.array

    # V_file.write(t.expression().value)

    t_ = t.expression().value
    t_prev_ = t_ - dt

    w_ex_final = (x[0]**2) * t_ + (x[1]**2) * t_ + (x[2]**2) * t_
    w_ex_prev = (x[0]**2) * t_prev_ + (x[1]**2) * t_prev_ + (x[2]**2) * t_prev_

    w_n_ex = (w_ex_final - w_ex_prev)/dt
    w_n_unique = (w_n - w_n_prev) /dt


t_fin= dt*n
t_prev = t_fin - dt

w_ex_final = (x[0]**2) * t_ + (x[1]**2) * t_ + (x[2]**2) * t_
w_ex_prev = (x[0]**2) * t_prev_ + (x[1]**2) * t_prev_ + (x[2]**2) * t_prev_

w_n_ex = (w_ex_final - w_ex_prev)/dt
w_n_unique = (w_n - w_n_prev) /dt

print(L2_norm(w_n_unique - w_n_ex))
