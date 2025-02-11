#%%
import numpy as np
from basix.ufl import element
from dolfinx import default_scalar_type
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
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, assemble_vector
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (
    FacetNormal,
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    cross,
    curl,
    diff,
    div,
    dot,
    grad,
    inner,
    variable,
    cos, 
    sin, 
    pi,
)
from dolfinx.common import Timer
from generate_team30_meshes_3D import domain_parameters, model_parameters
from utils import update_current_density
from utils2 import par_print


comm = MPI.COMM_WORLD

single_phase = True
ext = "single"
fname = f"meshes/{ext}_phase3D"
domains, currents = domain_parameters(single_phase)

with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    ct = xdmf.read_meshtags(mesh, name="Cell_markers")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    ft = xdmf.read_meshtags(mesh, name="Facet_markers")


degree = 1

num_phases = 1
steps_per_phase = 100
freq = model_parameters["freq"]
T = num_phases * 1 / freq
dt_ = 1.0 / steps_per_phase * 1 / freq

# Outer boundary domain
def boundary_marker(x):
    return np.full(x.shape[1], True)

mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = locate_entities_boundary(mesh, dim=tdim - 1, marker=boundary_marker)

# Define subdomains

Omega_n = domains["Cu"] + domains["Stator"] + domains["Air"] + domains["AirGap"]
Omega_c = domains["Rotor"] + domains["Al"]

# Set material parameters

DG0 = functionspace(mesh, ("DG", 0))
nu = Function(DG0)
sigma = Function(DG0)
density = Function(DG0)

mu_0 = model_parameters["mu_0"]
omega_J = 2 * np.pi * freq

for material, domain in domains.items():
    for marker in domain:
        cells = ct.find(marker)
        sigma.x.array[cells] = model_parameters["sigma"][material]
        nu.x.array[cells] = 1 / (model_parameters["mu_r"][material] * mu_0)

# RHS
t = 0.0
J0z = Function(DG0)
update_current_density(J0z, omega_J, t, ct, currents)
f1 = 0.0

# Solver

x = SpatialCoordinate(mesh)
dx = Measure("dx", domain=mesh, subdomain_data=ct)
dt = Constant(mesh, dt_)

nedelec_elem = element("N1curl", mesh.basix_cell(), degree)
V = functionspace(mesh, nedelec_elem)
lagrange_elem = element("Lagrange", mesh.basix_cell(), degree)
V1 = functionspace(mesh, lagrange_elem)

u_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
u1_dofs = V1.dofmap.index_map.size_global * V1.dofmap.index_map_bs
total_dofs = u_dofs + u1_dofs
par_print(comm, f"Total degrees of freedom: {total_dofs}")

# Initial conditions
u_n = Function(V)
u_n1 = Function(V1)

#Boundary conditions

bc_u = Function(V)
bc_u.x.array[:] = 0
boundary_dofs_u = locate_dofs_topological(V, entity_dim=tdim - 1, entities=boundary_facets)
bc0 = dirichletbc(bc_u, boundary_dofs_u)

bc_u1 = Function(V1)
bc_u1.x.array[:] = 0
boundary_dofs_u1 = locate_dofs_topological(V1, entity_dim=tdim - 1, entities=boundary_facets)
bc1 = dirichletbc(bc_u1, boundary_dofs_u1)

bc = [bc0, bc1]

# Define variational problem

u = TrialFunction(V)
v = TestFunction(V)

u1 = TrialFunction(V1)
v1 = TestFunction(V1)

a00 = dt * inner(nu * curl(u), curl(v)) * dx + inner((u * sigma), v) * dx

a01 = dt * inner(sigma * grad(u1), v) * dx
a10 = inner(sigma * u, grad(v1)) * dx

a11 = dt * inner(sigma * grad(u1), grad(v1)) * dx

L0 = dt * inner(J0z, v[2]) * dx + inner(sigma * u_n, v) * dx
L1 = inner(grad(v1), sigma * u_n) * dx

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
is_u1 = PETSc.IS().createStride(
    u1_map.size_local, offset_u1, 1, comm=PETSc.COMM_SELF
)

ksp = PETSc.KSP().create(mesh.comm)
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

V_CG = functionspace(mesh, ("CG", degree))._cpp_object
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
    pc0.setHYPRESetEdgeConstantVectors(
        cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec
    )
else:
    Vec_CG = functionspace(mesh, ("CG", degree, (mesh.geometry.dim,)))
    Pi = interpolation_matrix(Vec_CG._cpp_object, V._cpp_object)
    Pi.assemble()

    # Attach discrete gradient to preconditioner
    pc0.setHYPRESetInterpolations(mesh.geometry.dim, None, None, Pi, None)

opts = PETSc.Options()
opts[f"{ksp_u.prefix}pc_hypre_ams_cycle_type"] = 7
# opts[f"{ksp_u.prefix}pc_hypre_ams_tol"] = 0
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

u_n_prev = u_n.copy()

uh, uh1 = Function(V), Function(V1)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

sol = A_mat.createVecRight()

vector_vis = functionspace(
    mesh, ("Discontinuous Lagrange", degree + 1, (mesh.geometry.dim,))
)

da_dt = (u_n - u_n_prev) / dt
E = -grad(u_n1) - da_dt
B = curl(u_n)
J = sigma * E

A_vis = Function(vector_vis)
A_file = VTXWriter(mesh.comm, "A.bp", A_vis, "BP4")
A_vis.interpolate(u_n)
A_file.write(t)

B_vis = Function(vector_vis)
B_file = VTXWriter(mesh.comm, "B.bp", B_vis, "BP4")
Bexpr = Expression(B, vector_vis.element.interpolation_points())
B_vis.interpolate(Bexpr)
B_file.write(t)

V_file = VTXWriter(mesh.comm, "V.bp", u_n1, "BP4")
V_file.write(t)

J_vis = Function(vector_vis)
J_expr = Expression(J, vector_vis.element.interpolation_points())
J_vis.interpolate(J_expr)
J_file = VTXWriter(mesh.comm, "J.bp", J_vis, "BP4")

E_vis = Function(vector_vis)
E_expr = Expression(E, vector_vis.element.interpolation_points())
E_vis.interpolate(E_expr)
E_file = VTXWriter(mesh.comm, "E.bp", E_vis, "BP4")


num_steps = num_phases * steps_per_phase


for n in range(num_steps):
    t += dt_

    u_n_prev = u_n.copy()

    b = assemble_vector_block(L, a, bcs=bc)

    sol = A_mat.createVecRight()
    ksp.solve(b, sol)

    uh.x.array[:offset] = sol.array_r[:offset]
    uh1.x.array[: (len(sol.array_r) - offset)] = sol.array_r[offset:]

    uh.x.scatter_forward()
    uh1.x.scatter_forward()

    u_n.x.array[:] = uh.x.array
    u_n1.x.array[:] = uh1.x.array

    u_n.x.scatter_forward()
    u_n1.x.scatter_forward()

    B = curl(u_n)
    da_dt = (u_n - u_n_prev) / dt
    E = -grad(u_n1) - da_dt
    J = sigma * E

    if n % 10 == 0:
        A_vis.interpolate(u_n)
        A_file.write(t)

        V_file.write(t)

        B_vis.interpolate(Bexpr)
        B_file.write(t)

        E_expr = Expression(E, vector_vis.element.interpolation_points())
        E_vis.interpolate(E_expr)
        E_file.write(t)

        J_expr = Expression(J, vector_vis.element.interpolation_points())
        J_vis.interpolate(J_expr)
        J_file.write(t)



