# %%
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
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.io import VTXWriter
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
    MixedFunctionSpace,
)
from dolfinx.common import Timer
from dolfinx.io import XDMFFile
from utils import L2_norm, par_print, convert_facet_tags
from dolfinx.mesh import create_submesh
from generate_mesh import box_with_inner

comm = MPI.COMM_WORLD
degree = 1

ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 10  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

domain, ct, ft, vol_ids, boundary_ids = box_with_inner(comm, 0.1)


tdim = domain.topology.dim

x = SpatialCoordinate(domain)

t = variable(Constant(domain, ti))

sigma_value = 1
nu_value = 1

def exact(x, t):
    return as_vector(
        (
            cos(pi * x[1]) * sin(pi * t),
            cos(pi * x[2]) * sin(pi * t),
            cos(pi * x[0]) * sin(pi * t),
        )
    )


def exact1(x):
    return sin(pi * x[0]) * sin(pi * x[1]) * sin(pi * x[2])


uex = exact(x, t)
uex1 = exact1(x)


tdim = domain.topology.dim
fdim = tdim - 1

inner_domain = vol_ids["inner"]
outer_domain = vol_ids["outer"]
whole = (inner_domain, outer_domain)

inner_cells = ct.find(vol_ids["inner"])
outer_cells = ct.find(vol_ids["outer"])

submesh_inner, subdomain_inner_to_domain = create_submesh(domain, tdim, inner_cells)[:2]

cell_imap = domain.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts

mesh_to_submesh_inner = np.full(num_cells, -1)
mesh_to_submesh_inner[subdomain_inner_to_domain] = np.arange(
    len(subdomain_inner_to_domain)
)

entity_maps = {
    submesh_inner: mesh_to_submesh_inner,
}

dx = Measure("dx", domain, subdomain_data=ct)

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
V = functionspace(domain, nedelec_elem)
lagrange_elem = element("Lagrange", submesh_inner.basix_cell(), degree)
V1 = functionspace(submesh_inner, lagrange_elem)

domain_tags = ct
facet_tags = ft

dt = Constant(domain, d_t)
nu = Constant(domain, default_scalar_type(nu_value))
sigma = Constant(domain, default_scalar_type(sigma_value))

u = TrialFunction(V)
v = TestFunction(V)

u1 = TrialFunction(V1)
v1 = TestFunction(V1)

u_n = Function(V)
u_n1 = Function(V1)

a00 = dt * inner(nu * curl(u), curl(v)) * dx(whole) + inner((u * sigma), v) * dx(whole)

a01 = dt * inner(sigma * grad(u1), v) * dx(inner_domain)
a10 = inner(sigma * u, grad(v1)) * dx(whole)

a11 = dt * inner(sigma * grad(u1), grad(v1)) * dx(inner_domain)

a = form([[a00, a01], [a10, a11]], entity_maps=entity_maps)

f01 = nu * curl(curl(uex))
f02 = sigma * diff(uex, t) 
f03 = sigma * grad(uex1)

f11 = - div(sigma * diff(uex, t))
f12 = -div(sigma * grad(uex1)) 

L0 = (
    dt * inner(f01, v) * dx(whole)
    + dt * inner(f02, v) * dx(whole)
    + inner(sigma * u_n, v) * dx(whole)
    + dt * inner(f03, v) * dx(inner_domain)
)

L1 = dt * f11 * v1 * dx(whole) + dt * f12 * v1 * dx(inner_domain) + inner(grad(v1), sigma * u_n) * dx(whole)

L = form([L0, L1], entity_maps=entity_maps)

submesh_inner.topology.create_connectivity(fdim, tdim)
ft_inner = convert_facet_tags(domain, submesh_inner, subdomain_inner_to_domain, ft)
dofs_interface = locate_dofs_topological(V1, fdim, ft_inner.find(boundary_ids["inner"]))

u_bc_inner = Function(V1)
u_bc_expr_inner = Expression(uex1, V1.element.interpolation_points())
u_bc_inner.interpolate(u_bc_expr_inner)
bc_inner = dirichletbc(u_bc_inner, dofs_interface)


dofs_boundary = locate_dofs_topological(V, fdim, ft.find(boundary_ids["outer"]))

u_bc_outer = Function(V)
u_bc_expr_outer = Expression(uex, V.element.interpolation_points())
u_bc_outer.interpolate(u_bc_expr_outer)
bc_outer = dirichletbc(u_bc_outer, dofs_boundary)

bc = [bc_inner, bc_outer]

A = assemble_matrix_block(a, bcs=bc)
A.assemble()

b = assemble_vector_block(L, a, bcs=bc)


a_p = form([[a00, None], [None, a11]], entity_maps=entity_maps)
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
ksp.setOperators(A, P)
ksp.setType("gmres")
ksp.setTolerances(rtol=1e-10)
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
ksp.getPC().setFieldSplitIS(("u", is_u), ("u1", is_u1))
ksp_u, ksp_u1 = ksp.getPC().getFieldSplitSubKSP()

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
    pc0.setHYPRESetEdgeConstantVectors(
        cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec
    )
else:
    Vec_CG = functionspace(domain, ("CG", degree, (domain.geometry.dim,)))
    Pi = interpolation_matrix(Vec_CG._cpp_object, V._cpp_object)
    Pi.assemble()

    # Attach discrete gradient to preconditioner
    pc0.setHYPRESetInterpolations(domain.geometry.dim, None, None, Pi, None)

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

sol = A.createVecRight()

vector_vis = functionspace(
    domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,))
)

da_dt = (u_n - u_n_prev) / dt
E = -grad(u_n1) - da_dt
B = curl(u_n)
J = sigma * E


for n in range(num_steps):
    t.expression().value += d_t

    u_n_prev = u_n.copy()

    u_bc_inner.interpolate(u_bc_expr_inner)
    u_bc_outer.interpolate(u_bc_expr_outer)

    b = assemble_vector_block(L, a, bcs=bc)

    sol = A.createVecRight()
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
    iterations = ksp.getIterationNumber()
    print(ksp.getConvergedReason())



x_inner = SpatialCoordinate(domain)
uex_inner = exact(x_inner, t)
uex1_inner = exact1(x_inner)


t_prev = T - d_t

uex_prev_inner = exact(x_inner, t_prev)
uex_final_inner = exact(x_inner, T)

da_dt_exact = (uex_final_inner - uex_prev_inner) / d_t
E_exact = -grad(uex1) - da_dt_exact

B_exact = curl(uex)
#%%


# par_print(comm, f"E field error {L2_norm(E - E_exact)}")
par_print(comm, f"B field error {L2_norm(B - curl(uex))}")

iterations = ksp.getIterationNumber()

par_print(comm, f"Number of iterations: {iterations}")

# %%
