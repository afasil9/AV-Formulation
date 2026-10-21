import json

from mpi4py import MPI
from dolfinx import fem
from dolfinx.common import Timer
from dolfinx.fem import (
    Function,
    form,
)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
import numpy as np
from ufl import curl, inner, grad, diff, div, variable, cos, sin, pi
from basix.ufl import element
from petsc4py import PETSc
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from utils import L2_norm, par_print
import ufl
from dolfinx.fem import (
    functionspace,
    bcs_by_block,
    extract_function_spaces,
)
from utils import convert_facet_tags
from utils import interpolate_by_tags
from dolfinx.mesh import create_submesh
from dolfinx.io import XDMFFile
from dolfinx import default_scalar_type
from dolfinx.fem import assemble_scalar
from generate_mesh import create_box_with_sphere_msh, create_mesh_fenics

comm = MPI.COMM_WORLD
degree = 1

ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 10  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

mesh_type = "fenics"  # "fenics" (hex cube with inner box) or "gmsh" (box with sphere)

if mesh_type == "fenics":
    n = 20
    domain, ct, ft, vol_ids, boundary_ids = create_mesh_fenics(comm, n)
    mesh_file = "cube_with_inner_box.xdmf"
else:
    default_size = 0.1
    factor = 4
    cell_size = default_size / factor
    domain, ct, ft, vol_ids, boundary_ids = create_box_with_sphere_msh(comm, cell_size)
    mesh_file = "box_with_inner.xdmf"

with XDMFFile(domain.comm, mesh_file, "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(ct, domain.geometry)
    xdmf.write_meshtags(ft, domain.geometry)

tdim = domain.topology.dim
fdim = tdim - 1

par_print(comm, f"Number of cells in the domain: {domain.topology.index_map(tdim).size_global}")

x = ufl.SpatialCoordinate(domain)

t = variable(fem.Constant(domain, ti))
dt = fem.Constant(domain, d_t)

nu_value = 1


def exact(x, t):
    return ufl.as_vector(
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

inner_domain = vol_ids["inner"]
outer_domain = vol_ids["outer"]
whole = (inner_domain, outer_domain)

inner_cells = ct.find(inner_domain)

submesh_inner, subdomain_inner_to_domain = create_submesh(domain, tdim, inner_cells)[:2]
entity_maps = [subdomain_inner_to_domain]

dx = ufl.Measure("dx", domain=domain, subdomain_data=ct)

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
V = fem.functionspace(domain, nedelec_elem)

lagrange_elem = element("Lagrange", submesh_inner.basix_cell(), degree)
V1 = fem.functionspace(submesh_inner, lagrange_elem)

ndofs_V = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
nodfs_V1 = V1.dofmap.index_map.size_global * V1.dofmap.index_map_bs
total_dofs = ndofs_V + nodfs_V1

par_print(comm, f"Number of dofs in V (Nedelec): {ndofs_V}")
par_print(comm, f"Number of dofs in V1 (Lagrange): {nodfs_V1}")

nu = fem.Constant(domain, default_scalar_type(nu_value))

sigma_inner = fem.Constant(domain, default_scalar_type(1.0))
sigma_outer = fem.Constant(domain, default_scalar_type(0.0))

sigma_values = {
    inner_domain: sigma_inner,
    outer_domain: sigma_outer,
}

DG0 = fem.functionspace(domain, ("DG", 0))  # Piecewise constant function space

sigma = fem.Function(DG0)
interpolate_by_tags(sigma, sigma_values, ct)

#%%

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)

u_n = fem.Function(V)
u_n1 = fem.Function(V1)

a00 = dt * inner(nu * curl(u), curl(v)) * dx(whole) + inner(sigma * u, v) * dx(whole)

a01 = dt * inner(sigma * grad(u1), v) * dx(inner_domain)
a10 = inner(sigma * u, grad(v1)) * dx(inner_domain)

a11 = dt * inner(sigma * grad(u1), grad(v1)) * dx(inner_domain)

f01 = nu * curl(curl(uex))
f02 = sigma * diff(uex, t)
f03 = sigma * grad(uex1)

f11 = -div(sigma * diff(uex, t))
f12 = -div(sigma * grad(uex1))

L0 = (
    dt * inner(f01, v) * dx(whole)
    + dt * inner(f02, v) * dx(whole)
    + inner(sigma * u_n, v) * dx(whole)
    + dt * inner(f03, v) * dx(inner_domain)
)

L1 = (
    dt * f11 * v1 * dx(inner_domain)
    + dt * f12 * v1 * dx(inner_domain)
    + inner(grad(v1), sigma * u_n) * dx(inner_domain)
)

a = form([[a00, a01], [a10, a11]], entity_maps=entity_maps)
L = form([L0, L1], entity_maps=entity_maps)

# Boundary conditions

submesh_inner.topology.create_connectivity(fdim, tdim)
ft_inner = convert_facet_tags(submesh_inner, subdomain_inner_to_domain, ft)

dofs_interface = fem.locate_dofs_topological(V1, fdim, ft_inner.find(boundary_ids["interface"]))
u_bc_inner = fem.Function(V1)
u_bc_expr_inner = fem.Expression(uex1, V1.element.interpolation_points)
u_bc_inner.interpolate(u_bc_expr_inner)
bc_inner = fem.dirichletbc(u_bc_inner, dofs_interface)

dofs_boundary = fem.locate_dofs_topological(V, fdim, ft.find(boundary_ids["boundary"]))
u_bc_outer = fem.Function(V)
u_bc_expr_outer = fem.Expression(uex, V.element.interpolation_points)
u_bc_outer.interpolate(u_bc_expr_outer)
bc_outer = fem.dirichletbc(u_bc_outer, dofs_boundary)

bcs = [bc_inner, bc_outer]

# Solver steps
par_print(comm, "Assembling system")

with Timer("Setup: Assembling system") as a_mat_timer:
    A_mat = assemble_matrix(a, bcs=bcs)
    A_mat.assemble()

    a_p = form([[a00, None], [None, a11]], entity_maps=entity_maps)
    P = assemble_matrix(a_p, bcs=bcs)
    P.assemble()
assembly_time = a_mat_timer.elapsed().total_seconds()

bcs1 = bcs_by_block(extract_function_spaces(a, 1), bcs)
bcs0 = bcs_by_block(extract_function_spaces(L), bcs)

u_map = V.dofmap.index_map
u1_map = V1.dofmap.index_map

offset_u = u_map.local_range[0] * V.dofmap.index_map_bs + u1_map.local_range[0]
offset_u1 = offset_u + u_map.size_local * V.dofmap.index_map_bs

is_u = PETSc.IS().createStride(
    u_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=domain.comm
)
is_u1 = PETSc.IS().createStride(u1_map.size_local, offset_u1, 1, comm=domain.comm)

par_print(comm, "Setting up preconditioner")
timer_pc = Timer("Setup: Preconditioner")

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A_mat, P)
ksp.setType("gmres")
ksp.setTolerances(rtol=1e-10)
ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
ksp.getPC().setFieldSplitIS(("u", is_u), ("u1", is_u1))
ksp_u, ksp_u1 = ksp.getPC().getFieldSplitSubKSP()

ksp_u.setType("preonly")
ksp_u.getPC().setType("hypre")
ksp_u.getPC().setHYPREType("ams")

W = fem.functionspace(domain, ("Lagrange", degree))
G = discrete_gradient(W._cpp_object, V._cpp_object)
G.assemble()

# Remove numerically zero entries so that G has exactly 2 entries per row
ai, aj, av = G.getValuesCSR()
keep = np.abs(av) > 1e-12
row_ids = np.repeat(np.arange(len(ai) - 1), np.diff(ai))[keep]
indptr = np.concatenate(([0], np.cumsum(np.bincount(row_ids, minlength=len(ai) - 1))))
G_clean = PETSc.Mat().createAIJ(
    size=G.getSizes(),
    csr=(indptr.astype(PETSc.IntType), aj[keep], av[keep]),
    comm=G.comm,
)
G_clean.assemble()
ksp_u.getPC().setHYPREDiscreteGradient(G_clean)

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
    ksp_u.getPC().setHYPRESetEdgeConstantVectors(
        cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec
    )

else:
    shape = (domain.geometry.dim,)
    Q = fem.functionspace(domain, ("Lagrange", degree, shape))
    Pi = interpolation_matrix(Q._cpp_object, V._cpp_object)
    Pi.assemble()
    ksp_u.getPC().setHYPRESetInterpolations(dim=domain.geometry.dim, ND_Pi_Full=Pi)

opts = PETSc.Options()
opts[f"{ksp_u.prefix}pc_hypre_ams_cycle_type"] = 13
opts[f"{ksp_u.prefix}pc_hypre_ams_tol"] = 0.0
opts[f"{ksp_u.prefix}pc_hypre_ams_max_iter"] = 1
opts[f"{ksp_u.prefix}pc_hypre_ams_print_level"] = 0
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_alpha_options"] = "10,1,6,6,4"
opts[f"{ksp_u.prefix}pc_hypre_ams_amg_beta_options"] = "10,1,6,6,4"
opts[f"{ksp_u.prefix}pc_hypre_ams_projection_frequency"] = 100
opts[f"{ksp_u.prefix}pc_hypre_ams_relax_type"] = 2
opts[f"{ksp_u.prefix}pc_hypre_ams_relax_weight"] = 1.0
opts[f"{ksp_u.prefix}pc_hypre_ams_relax_times"] = 1
opts[f"{ksp_u.prefix}pc_hypre_ams_omega"] = 1.0

V_interior = fem.functionspace(domain, ("CG", degree))
interior_nodes_array = fem.Function(V_interior)

interior_nodes_array.x.array[:] = 1.0
interior_nodes_array.x.scatter_forward()

dofmap = W.dofmap
num_dofs_per_cell = dofmap.dof_layout.num_dofs
cell_dofs = dofmap.list.reshape(-1, num_dofs_per_cell)

tags = (inner_domain,)
tagged_cells = np.unique(np.concatenate([ct.find(tag) for tag in tags]))

tagged_cell_dofs = cell_dofs[tagged_cells].flatten()
unique_dofs = np.unique(tagged_cell_dofs)

interior_nodes_array.x.array[unique_dofs] = 0.0

# Exclude the outer domain boundary from the interior node set
boundary_dofs_interior = fem.locate_dofs_topological(V_interior, fdim, ft.find(boundary_ids["boundary"]))
interior_nodes_array.x.array[boundary_dofs_interior] = 0.0
interior_nodes_array.x.scatter_forward()

ksp_u.getPC().setHYPREAMSSetInteriorNodes(interior_nodes_array.x.petsc_vec)

ksp_u.setFromOptions()

ksp_u1.setType("preonly")
ksp_u1.getPC().setType("gamg")

ksp.setUp()
ksp_u.getPC().setUp()
ksp_u1.getPC().setUp()

timer_pc.stop()
pc_assembly_time = timer_pc.elapsed().total_seconds()

uh, uh1 = Function(V), Function(V1)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

u_n_prev = fem.Function(V)

assembly_time_rhs = []
solve_time = []
iterations = []

for n in range(num_steps):

    t.expression().value += d_t

    par_print(comm, f"Time step {n+1}: t = {t.expression().value}")

    u_n_prev.x.array[:] = u_n.x.array[:]

    u_bc_inner.interpolate(u_bc_expr_inner)
    u_bc_outer.interpolate(u_bc_expr_outer)

    with Timer("Assembling RHS") as rhs_timer:
        b = assemble_vector(L)
        apply_lifting(b, a, bcs=bcs1)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs0)
    assembly_time_rhs.append(rhs_timer.elapsed().total_seconds())

    sol = A_mat.createVecRight()

    with Timer("Solve") as timer_solve:
        ksp.solve(b, sol)
    solve_time.append(timer_solve.elapsed().total_seconds())
    iterations.append(ksp.getIterationNumber())

    uh.x.array[:offset] = sol.array_r[:offset]
    uh1.x.array[:(len(sol.array_r) - offset)] = sol.array_r[offset:]

    uh.x.scatter_forward()
    uh1.x.scatter_forward()

    u_n.x.array[:] = uh.x.array
    u_n1.x.array[:] = uh1.x.array

    u_n.x.scatter_forward()
    u_n1.x.scatter_forward()

    par_print(comm, f"KSP converged with reason {ksp.getConvergedReason()}")

#%%

# Error computation

B = curl(u_n)
da_dt = (u_n - u_n_prev) / dt
E = -grad(u_n1) - da_dt

t_prev = T - d_t

uex_prev = exact(x, t_prev)
uex_final = exact(x, T)

da_dt_exact = (uex_final - uex_prev) / d_t
E_exact = -grad(uex1) - da_dt_exact

B_exact = curl(uex)

E_err = E - E_exact
B_err = B - B_exact

err = assemble_scalar(
    form(inner(E_err, E_err) * dx(inner_domain), entity_maps=entity_maps)
)
E_field_error = np.sqrt(comm.allreduce(err, op=MPI.SUM))

par_print(comm, f"E field error {E_field_error}")
par_print(comm, f"B field error {L2_norm(B_err)}")
par_print(comm, f"Number of iterations: {ksp.getIterationNumber()}")

timings = {
    "mesh_type": mesh_type,
    "total_dofs": total_dofs,
    "num_steps": num_steps,
    "assemble_matrix": assembly_time,
    "assemble_preconditioner": pc_assembly_time,
    "assemble_rhs": assembly_time_rhs,
    "assemble_rhs_total": sum(assembly_time_rhs),
    "solve": solve_time,
    "solve_total": sum(solve_time),
    "iterations": iterations,
}

if comm.rank == 0:
    timings_file = f"timings_mixed_{mesh_type}.json"
    print(f"Saving timings to '{timings_file}'.")
    with open(timings_file, "w") as f:
        json.dump(timings, f, indent=4)
