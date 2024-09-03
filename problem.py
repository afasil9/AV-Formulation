import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (ds, dx, inner, grad, div, curl, SpatialCoordinate, 
                 as_vector, sin, cos, pi, variable, TrialFunction, 
                 TestFunction, Measure, diff)
from dolfinx import fem, mesh, io, default_scalar_type
from dolfinx.fem import (dirichletbc, assemble_scalar, form, Function, 
                         Expression, locate_dofs_topological)
from dolfinx.fem.petsc import (LinearProblem, assemble_matrix_block, 
                               assemble_vector_block)
from dolfinx.io import VTXWriter
from basix.ufl import element
from ufl.core.expr import Expr

iteration_count = []
residual_norm = []

n = 30
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
te = 0.01

dt = fem.Constant(domain, te / num_steps)

t = variable(fem.Constant(domain, 0.0))

# Should be topology dim
gdim = domain.geometry.dim
facet_dim = gdim - 1

# Define function spaces
nedelec_elem = element("N1curl", domain.basix_cell(), degree)
A_space = fem.functionspace(domain, nedelec_elem)

lagrange_element = element("Lagrange", domain.basix_cell(), degree)
V_space = fem.functionspace(domain, lagrange_element)

# Magnetic Vector Potential
A  = TrialFunction(A_space)
v = TestFunction(A_space)

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
    return as_vector((cos(pi * x[1]) * sin(pi * t), cos(pi * x[2]) * sin(pi * t), cos(pi * x[0]) * sin(pi * t)))

def V_ex(x, t):
    return sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2]) * sin(pi * t)

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

A_mat = assemble_matrix_block(a, bcs = bc)
A_mat.assemble()

# Need to interpolate if non-zero initially
A_n = Function(A_space)
S_n = Function(V_space)

print(f"A norm = {A_mat.norm()}")

j_e = (1 / mu_R) * curl(curl(aex)) + sigma*diff(aex,t)+ sigma*grad(diff(vex,t)) # Strong Form
time_l0 = (sigma * A_n + sigma * grad(S_n))
f_time_l0 = dt * j_e + time_l0

f_l1 = inner(-dt * div(j_e), q)  # Strong Form
time_l1 = inner(grad(q), sigma*A_n) + inner(grad(q), sigma*grad(S_n))
f_time_l1 = f_l1 + time_l1

L0 = inner(f_time_l0,v) * dx
L1 = f_time_l1 * dx             

L = form([L0, L1])

b = assemble_vector_block(L, a, bcs = bc)
print(f"b norm = {b.norm()}")

A_map = A_space.dofmap.index_map
V_map = V_space.dofmap.index_map

offset_A = A_map.local_range[0] * A_space.dofmap.index_map_bs + V_map.local_range[0]
offset_S = offset_A + A_map.size_local * A_space.dofmap.index_map_bs
is_A = PETSc.IS().createStride(A_map.size_local * A_space.dofmap.index_map_bs, offset_A, 1, comm=PETSc.COMM_SELF)
is_S = PETSc.IS().createStride(V_map.size_local, offset_S, 1, comm=PETSc.COMM_SELF)

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

offset = A_space.dofmap.index_map.size_local * A_space.dofmap.index_map_bs

aerr = []
serr = []
res = []

X = fem.functionspace(domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,)))
A_vis = fem.Function(X)
A_vis.interpolate(A_n)

A_file = io.VTXWriter(domain.comm, "A.bp", A_vis, "BP4")
A_file.write(t.expression().value)

V_file = io.VTXWriter(domain.comm, "V.bp", S_n, "BP4")
V_file.write(t.expression().value)

for i in range(num_steps):  
    print(f"Step {i + 1} of {num_steps}")

    t.expression().value += dt.value

    u_bc_A.interpolate(u_bc_expr_A)
    u_bc_V.interpolate(u_bc_expr_V)
    
    # with b.localForm() as loc_b:
    #     loc_b.set(0)

    # Better to not create b and sol each time step
    b = assemble_vector_block(L, a, bcs = bc)
    sol = A_mat.createVecRight()  #Solution Vector    
    ksp.solve(b, sol)
       
    residual = A_mat * sol - b
    # print('residual is ', residual.norm())
    res.append(residual.norm())

    A_n.x.array[:offset] = sol.array_r[:offset]
    S_n.x.array[:(len(sol.array_r) - offset)] = sol.array_r[offset:]

    A_vis.interpolate(A_n)
    A_file.write(t.expression().value)
    V_file.write(t.expression().value)


A_file.close()

def dV_ex_dt(x, t):
    return pi * cos(pi * t) * sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2])


print(f"e_B  = {L2_norm(curl(A_n) - curl(aex))}")
# print(f"e_V  = {L2_norm(S_n - vex)}")
dvex_dt = dV_ex_dt(x, t)
print(f"e_V = {L2_norm(S_n - dvex_dt)}")
