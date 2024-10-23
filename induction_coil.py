#%%
from mpi4py import MPI
import numpy
import ufl
from petsc4py import PETSc
from dolfinx import fem, default_scalar_type, io
from dolfinx.fem import functionspace, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, apply_lifting, set_bc
from ufl import SpatialCoordinate, sin, pi, grad, div, variable, diff, cos, curl, ds
from dolfinx.fem import Function, Expression, dirichletbc, form
import numpy as np
from basix.ufl import element
import ufl.constant
from ufl.core.expr import Expr
from scipy.linalg import norm
from dolfinx.io import XDMFFile
from dolfinx.fem import locate_dofs_topological
from ufl import Measure

with XDMFFile(MPI.COMM_WORLD, "em_model2_refined.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="domains")
    domain_tags = xdmf.read_meshtags(domain, name="domains")
    ftmesh = xdmf.read_mesh(name="facets")
    ft = xdmf.read_meshtags(ftmesh, name="facets")
    domain.topology.create_connectivity(2, 3)


ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 2  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

degree = 1

t = variable(fem.Constant(domain, ti))
dt = fem.Constant(domain, d_t)

dx = Measure("dx", domain=domain, subdomain_data=domain_tags)

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
V = functionspace(domain, nedelec_elem)

lagrange_element = element("Lagrange", domain.basix_cell(), degree)
V1 = functionspace(domain, lagrange_element)

const = fem.functionspace(domain, ("DG", 0)) #Piecewise constant function space

# Define material properties
mu_0 = 1.2566e-09 
mu_r = fem.Constant(domain, default_scalar_type(1.0))
mu_c = fem.Constant(domain, default_scalar_type(1.0))

sigma_air = fem.Constant(domain, default_scalar_type(1e-8))
sigma_copper = fem.Constant(domain, default_scalar_type(5.96e4))

sigma = fem.Function(const)
mu = fem.Function(const)
nu = fem.Function(const)

def interpolate_by_tags(function, value_dict, domain_tags):
    for tag, value in value_dict.items():
        function.interpolate(lambda x: np.full_like(x[0], value), domain_tags.find(tag))
    function.x.scatter_forward()

sigma_values = {
    1: sigma_air,
    2: sigma_air,
    3: sigma_copper,
    4: sigma_air
}

mu_values = {
    1: mu_0 * mu_r,
    2: mu_0 * mu_r,
    3: mu_0 * mu_c,
    4: mu_0 * mu_r
}

nu_values = {
    1: 1 / (mu_0 * mu_r),
    2: 1 / (mu_0 * mu_r),
    3: 1 / (mu_0 * mu_c),
    4: 1 / (mu_0 * mu_r)
}

interpolate_by_tags(sigma, sigma_values, domain_tags)
interpolate_by_tags(nu, nu_values, domain_tags)

x = SpatialCoordinate(domain)
fdim = domain.topology.dim - 1

zeroA = fem.Function(V)    
zeroA.x.array[:] = 0
tags_to_find = [12,13,14,15,16,18]
boundary_entities = np.concatenate([ft.find(tag) for tag in tags_to_find])
bdofs0 = locate_dofs_topological(V, fdim, boundary_entities)
bc0 = dirichletbc(zeroA, bdofs0)

zeroV = fem.Constant(domain, PETSc.ScalarType(0.0)) 
sink = ft.find(9)
bdofs_low = locate_dofs_topological(V1, fdim, sink)
bc_low = dirichletbc(zeroV, bdofs_low, V1)

V_high = fem.Constant(domain, PETSc.ScalarType(1.0))
source = ft.find(10)
bdofs_high = locate_dofs_topological(V1, fdim, source)
bc_high = dirichletbc(V_high, bdofs_high, V1)

bdofs1 = locate_dofs_topological(V1, fdim, boundary_entities)
bc1 = dirichletbc(zeroV, bdofs1, V1)

bc = [bc0, bc1, bc_low, bc_high]

# Initial Conditions

u_n = fem.Function(V)
u_n1 = fem.Function(V1)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

j_sur = fem.Constant(domain, PETSc.ScalarType(1.0))

a00 = dt*nu*ufl.inner(curl(u), curl(v)) * dx + sigma*ufl.inner(u, v) * dx
L0 = sigma*ufl.inner(u_n, v) * dx 

u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)

f1 = fem.Constant(domain, PETSc.ScalarType(0))
a11 = ufl.inner(sigma*ufl.grad(u1), ufl.grad(v1)) * dx
L1 = f1 * v1 * dx + sigma*ufl.inner(grad(v1), u_n) * dx 

a01 = dt * ufl.inner(sigma*grad(u1), v) * dx
a10 = ufl.inner(sigma*grad(v1), u) * dx

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

offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

sol = A_mat.createVecRight()

X = fem.functionspace(domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,)))
V_scalar = fem.functionspace(domain, ("Discontinuous Lagrange", degree + 1))

A_vis = fem.Function(X)
A_vis.interpolate(u_n)

A_file = io.VTXWriter(domain.comm, "A.bp", A_vis, "BP4")
A_file.write(t.expression().value)

B_vis = fem.Function(X)
B_3D = curl(u_n)
Bexpr = fem.Expression(B_3D, X.element.interpolation_points())
B_vis.interpolate(Bexpr)

B_file = io.VTXWriter(domain.comm, "B.bp", B_vis, "BP4")
B_file.write(t.expression().value)

V_file = io.VTXWriter(domain.comm, "V.bp", u_n1, "BP4")
V_file.write(t.expression().value)

J_vis = fem.Function(X)
J_file = io.VTXWriter(domain.comm, "J.bp", J_vis, "BP4")
J_file.write(t.expression().value)

for n in range(num_steps):
    t.expression().value += d_t
    print(n)
    
    b = assemble_vector_block(L, a, bcs=bc)

    u_n_prev = u_n.copy()

    sol = A_mat.createVecRight()
    ksp.solve(b, sol)

    residual = A_mat * sol - b
    print('residual is ', residual.norm())

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

