#%%
from mpi4py import MPI
from ufl import SpatialCoordinate, variable, as_vector, grad, curl, div, diff, sin, cos, pi
from dolfinx import fem
from utils import L2_norm, create_mesh_fenics, create_mesh_gmsh
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import default_scalar_type
from dolfinx.fem import dirichletbc,form, Function, Expression, locate_dofs_topological, functionspace, Constant
from dolfinx.fem.petsc import assemble_vector_block, assemble_matrix_block
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.io import VTXWriter
from basix.ufl import element
from ufl import grad, inner, curl, Measure, Constant, TrialFunction, TestFunction, variable, div, FacetNormal
from dolfinx import fem
import ufl
from utils import L2_norm

comm = MPI.COMM_WORLD
degree = 1
n = 4

ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 100  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

boundaries = {"bottom": 1, "top": 2, "front": 3, "right": 4, "back": 5, "left": 6}
domain,ft, ct = create_mesh_fenics(comm, n, boundaries)

x = SpatialCoordinate(domain)
t = variable(fem.Constant(domain, ti))
nu_ = 1.0
sigma_ = 1.0

# def exact(x, t):
#     return as_vector((cos(pi * x[1]) * sin(pi * t), cos(pi * x[2]) * sin(pi * t), cos(pi * x[0]) * sin(pi * t)))

# def exact1(x):
#     return sin(pi*x[0]) * sin(pi*x[1]) * sin(pi*x[2])

def exact(x, t):
    return as_vector((
        x[1]**2 + x[0] * t, 
        x[2]**2 + x[1] * t, 
        x[0]**2 + x[2] * t))

def exact1(x):
    return (x[0]**2) + (x[1]**2) + (x[2]**2)

uex = exact(x,t)
uex1 = exact1(x)

f0 = nu_*curl(curl(uex)) + sigma_*diff(uex,t) + sigma_*grad(uex1)
f1 = -div(sigma_*grad(uex1)) - div(sigma_*diff(uex,t))

bc_dict = {
    "V": {
        # Interpolate vector-valued exact solution on boundaries
        (boundaries["bottom"], boundaries["top"], 
        boundaries["front"], boundaries["right"],
        boundaries["back"], boundaries["left"]): uex
    },
    "V1": {
        # Interpolate scalar exact solution on all boundaries
        (boundaries["bottom"], boundaries["top"], 
        boundaries["front"], boundaries["right"],
        boundaries["back"], boundaries["left"]): uex1
    }
}

preconditioner = 'AMS'

domain_tags = ct

#Solver

dx = Measure("dx", domain=domain, subdomain_data=domain_tags)
dt = fem.Constant(domain, d_t)
nu = fem.Constant(domain, default_scalar_type(nu_))
sigma = fem.Constant(domain, default_scalar_type(sigma_))

gdim = domain.geometry.dim
facet_dim = gdim - 1

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
V = fem.functionspace(domain, nedelec_elem)
lagrange_elem = element("Lagrange", domain.basix_cell(), degree)
V1 = fem.functionspace(domain, lagrange_elem)

u_n = Function(V)
u_expr = Expression(uex, V.element.interpolation_points())
u_n.interpolate(u_expr)

u_n1 = fem.Function(V1)
uex_expr1 = Expression(uex1, V1.element.interpolation_points())
u_n1.interpolate(uex_expr1)

bc0_list = []
bc1_list = []

for space, bc_dict_space in bc_dict.items():
    for tags, value in bc_dict_space.items():
        boundary_entities = np.concatenate([ft.find(tag) for tag in tags])

        if space == "V":
            # For Nédélec elements (vector field)
            bdofs = locate_dofs_topological(V, facet_dim, boundary_entities)
            u_bc_V = Function(V)
            u_expr_V = Expression(
                value, V.element.interpolation_points(), comm=MPI.COMM_SELF
            )
            u_bc_V.interpolate(u_expr_V)
            bc0_list.append(dirichletbc(u_bc_V, bdofs))

        elif space == "V1":
            # For Lagrange elements (scalar field)
            bdofs = locate_dofs_topological(V1, facet_dim, boundary_entities)
            u_bc_V1 = Function(V1)
            u_expr_V1 = Expression(
                value, V1.element.interpolation_points(), comm=MPI.COMM_SELF
            )
            u_bc_V1.interpolate(u_expr_V1)
            bc1_list.append(dirichletbc(u_bc_V1, bdofs))

bc = bc0_list + bc1_list

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)

a00 = dt * nu * ufl.inner(curl(u), curl(v)) * dx + sigma * ufl.inner(u, v) * dx
L0 = dt * ufl.inner(f0, v) * dx + sigma * ufl.inner(u_n, v) * dx

a01 = dt * sigma * ufl.inner(grad(u1), v) * dx
a10 = sigma * ufl.inner(grad(v1), u) * dx

a11 = dt * ufl.inner(sigma * ufl.grad(u1), ufl.grad(v1)) * dx
L1 = dt * f1 * v1 * dx + sigma * ufl.inner(grad(v1), u_n) * dx

a = form([[a00, a01], [a10, a11]])

A_mat = assemble_matrix_block(a, bcs=bc)
A_mat.assemble()

L = form([L0, L1])
b = assemble_vector_block(L, a, bcs=bc)

if preconditioner == "Direct":
    print("Direct solve")
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A_mat)
    ksp.setType("preonly")

    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")

    opts = PETSc.Options()
    opts["mat_mumps_icntl_14"] = 80
    opts["mat_mumps_icntl_24"] = 1
    opts["mat_mumps_icntl_25"] = 0
    opts["ksp_error_if_not_converged"] = 1
    ksp.setFromOptions()
else:
    print("AMS preconditioner")
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
            cvec_0.vector, cvec_1.vector, cvec_2.vector
        )
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

u_n_prev = u_n.copy()

uh, uh1 = Function(V), Function(V1)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

sol = A_mat.createVecRight()
ksp.solve(b, sol)

uh.x.array[:] = sol.array_r[:offset]
uh1.x.array[:] = sol.array_r[offset:]

u_n.x.array[:] = uh.x.array
u_n1.x.array[:] = uh1.x.array

print(ksp.getTolerances())

vector_vis = functionspace(domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,)))

da_dt = (u_n - u_n_prev) / dt
E = -grad(u_n1) - da_dt
B = curl(u_n)
J = sigma * E

A_vis = Function(vector_vis)
A_file = VTXWriter(domain.comm, "A.bp", A_vis, "BP4")
A_vis.interpolate(u_n)
A_file.write(t.expression().value)

B_vis = Function(vector_vis)
B_file = VTXWriter(domain.comm, "B.bp", B_vis, "BP4")
Bexpr = Expression(B, vector_vis.element.interpolation_points())
B_vis.interpolate(Bexpr)
B_file.write(t.expression().value)

V_file = VTXWriter(domain.comm, "V.bp", u_n1, "BP4")
V_file.write(t.expression().value)

J_vis = Function(vector_vis)
J_expr = Expression(J, vector_vis.element.interpolation_points())
J_vis.interpolate(J_expr)
J_file = VTXWriter(domain.comm, "J.bp", J_vis, "BP4")

E_vis = Function(vector_vis)
E_expr = Expression(E, vector_vis.element.interpolation_points())
E_vis.interpolate(E_expr)
E_file = VTXWriter(domain.comm, "E.bp", E_vis, "BP4")

#%%
for n in range(num_steps):

    t.expression().value += d_t

    u_n_prev = u_n.copy()

    u_bc_V.interpolate(u_expr_V)
    u_bc_V1.interpolate(u_expr_V1)

    b = assemble_vector_block(L, a, bcs=bc)

    sol = A_mat.createVecRight()
    ksp.solve(b, sol)

    uh.x.array[:] = sol.array_r[:offset]
    uh1.x.array[:] = sol.array_r[offset:]

    u_n.x.array[:] = uh.x.array
    u_n1.x.array[:] = uh1.x.array

    u_n.x.scatter_forward()
    u_n1.x.scatter_forward()

    if n % 10 == 0:
        print(f"Time step {n}")
        A_vis.interpolate(u_n)
        A_file.write(t.expression().value)

        V_file.write(t.expression().value)

        B = curl(u_n)
        B_vis.interpolate(Bexpr)
        B_file.write(t.expression().value)

        da_dt = (u_n - u_n_prev) / dt
        E = -grad(u_n1) - da_dt
        E_expr = Expression(E, vector_vis.element.interpolation_points())
        E_vis.interpolate(E_expr)
        E_file.write(t.expression().value)

        J = sigma * E
        J_expr = Expression(J, vector_vis.element.interpolation_points())
        J_vis.interpolate(J_expr)
        J_file.write(t.expression().value)

#%%
da_dt = (u_n - u_n_prev) / dt
E = -grad(u_n1) - da_dt
B = curl(u_n)
J = sigma * E

# print("norm of B", L2_norm(B))
# print("norm of B_vis", L2_norm(B_vis))
# print("norm of E", L2_norm(E))
# print("norm of E_vis", L2_norm(E_vis))
# print("norm of J", L2_norm(J))
# print("norm of J_vis", L2_norm(J_vis))

# Post pro

t_prev = T - d_t

uex_prev = exact(x, t_prev)
uex_final = exact(x, T)

da_dt_exact = (uex_final - uex_prev) / d_t
E_exact = -grad(uex1) - da_dt_exact

print("E field error", L2_norm(E - E_exact))
print("B field error", L2_norm(B - curl(uex)))