
from mpi4py import MPI
import numpy as np
import scipy.linalg as sla
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.mesh import create_unit_cube, locate_entities_boundary
from dolfinx.fem import Function, dirichletbc, locate_dofs_topological, form
from dolfinx.fem.petsc import assemble_matrix, discrete_gradient
from ufl import TrialFunction, TestFunction, curl, grad, inner, dx
from basix.ufl import element
from utils import boundary_marker, par_print

comm = MPI.COMM_WORLD

degree = 1

n = 8

domain = create_unit_cube(MPI.COMM_WORLD, n, n, n)
gdim = domain.geometry.dim
facet_dim = gdim - 1

DG = fem.functionspace(domain, ("DG", 0))
alpha = Function(DG)
alpha.interpolate(lambda x: np.where(x[0] <= 0.5, 7.96e5, 1.0))
beta = Function(DG)
beta.interpolate(lambda x: np.where(x[0] <= 0.5, 1e10, 1.0))

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
A_space = fem.functionspace(domain, nedelec_elem)
V_CG = fem.functionspace(domain, ("CG", degree))

A = TrialFunction(A_space)
v = TestFunction(A_space)
V = TrialFunction(V_CG)
q = TestFunction(V_CG)

a_AA = form(inner(alpha * curl(A), curl(v)) * dx + inner(beta * A, v) * dx)
a_AV = form(inner(beta * grad(V), v) * dx)
a_VA = form(inner(beta * A, grad(q)) * dx)
a_VV = form(inner(beta * grad(V), grad(q)) * dx)
a_curl = form(inner(alpha * curl(A), curl(v)) * dx)
a_mass = form(inner(beta * A, v) * dx)


facets = locate_entities_boundary(
    domain, dim=(domain.topology.dim - 1), marker=boundary_marker
)
dofs_A = locate_dofs_topological(V=A_space, entity_dim=facet_dim, entities=facets)
dofs_V = locate_dofs_topological(V=V_CG, entity_dim=facet_dim, entities=facets)
bc_A = dirichletbc(Function(A_space), dofs_A)
bc_V = dirichletbc(Function(V_CG), dofs_V)
bcs = [bc_A, bc_V]

A_mat = assemble_matrix([[a_AA, a_AV], [a_VA, a_VV]], bcs=bcs, kind="nest")
A_mat.assemble()
P_mat = assemble_matrix([[a_AA, None], [None, a_VV]], bcs=bcs, kind="nest")
P_mat.assemble()


def dense(mat):
    mat.assemble()
    return mat.convert("dense", PETSc.Mat()).getDenseArray().copy()


G = discrete_gradient(V_CG, A_space)
G.assemble()
G_d = dense(G)
curl_d = dense(assemble_matrix(a_curl))
mass_d = dense(assemble_matrix(a_mass))
AA_d = dense(assemble_matrix(a_AA))
VV_d = dense(assemble_matrix(a_VV))
free_A = np.setdiff1d(np.arange(AA_d.shape[0]), dofs_A)
free_V = np.setdiff1d(np.arange(VV_d.shape[0]), dofs_V)
AV_free = (mass_d @ G_d)[np.ix_(free_A, free_V)]
AA_free = AA_d[np.ix_(free_A, free_A)]
VV_free = VV_d[np.ix_(free_V, free_V)]

scale = np.abs(AA_d).max()
par_print(comm, "--- structural identities (relative to max|A~|) ---")
par_print(comm, f"||K_nu G||            = {np.abs(curl_d @ G_d).max() / scale:.2e}")
par_print(comm, f"||A~ G - M G||        = {np.abs(AA_d @ G_d - mass_d @ G_d).max() / scale:.2e}")
par_print(comm, f"||C - G^T M G||       = {np.abs(VV_d - G_d.T @ mass_d @ G_d).max() / scale:.2e}")
S = VV_free - AV_free.T @ np.linalg.solve(AA_free, AV_free)
par_print(comm, f"||S|| / ||C||         = {np.abs(S).max() / np.abs(VV_free).max():.2e}  (Schur complement)")

A_d, P_d = dense(A_mat), dense(P_mat)
lam = sla.eigh(A_d, P_d, eigvals_only=True)
ne, nv = len(free_A), len(free_V)
nbc = len(dofs_A) + len(dofs_V)  
tol = 1e-8
counts = [np.sum(np.abs(lam - t) < tol) for t in (0.0, 1.0, 2.0)]
par_print(comm, "--- generalised spectrum of (A_mat, P_mat) ---")
par_print(comm, f"ne_free = {ne}, nv_free = {nv}, bc rows = {nbc}, total = {len(lam)}")
par_print(comm, f"#lambda=0: {counts[0]:5d}   predicted nv         = {nv}")
par_print(comm, f"#lambda=1: {counts[1]:5d}   predicted ne-nv+nbc  = {ne - nv + nbc}")
par_print(comm, f"#lambda=2: {counts[2]:5d}   predicted nv         = {nv}")
par_print(comm, f"unaccounted eigenvalues: {len(lam) - sum(counts)}")
par_print(comm, f"max distance to {{0,1,2}}: {np.min(np.abs(lam[:, None] - [0, 1, 2]), axis=1).max():.2e}")
nz = lam[lam > tol]
par_print(comm, f"kappa_eff = {nz.max() / nz.min():.12f}")

T = np.linalg.solve(P_d, A_d)
I = np.eye(len(T))
x = np.random.default_rng(0).standard_normal(len(T))
y = T @ ((T - I) @ ((T - 2 * I) @ x))
par_print(comm, "--- minimal polynomial ---")
par_print(comm, f"||T(T-I)(T-2I)x|| / ||x|| = {np.linalg.norm(y) / np.linalg.norm(x):.2e}")
par_print(comm, f"||(T-I)(T-2I)x||  / ||x|| = {np.linalg.norm((T - I) @ ((T - 2 * I) @ x)) / np.linalg.norm(x):.2e}  (should NOT be small)")

is_A, is_V = A_mat.getNestISs()[0]


def run(ksp_type):
    fieldsplit_opts = {
        "ksp_type": ksp_type,
        "ksp_rtol": 1e-12,
        "ksp_atol": 1e-50,
        "ksp_max_it": 20,
        "ksp_monitor_true_residual": None,
        "ksp_view_eigenvalues": None,  # Ritz values of P_mat^{-1} A_mat from the Krylov space
        "pc_fieldsplit_type": "additive",  # block diagonal P_mat; schur is undefined (S = 0)
    }
    for s in ("A", "V"):
        fieldsplit_opts[f"fieldsplit_{s}_ksp_type"] = "preonly"
        fieldsplit_opts[f"fieldsplit_{s}_pc_type"] = "lu"
        fieldsplit_opts[f"fieldsplit_{s}_pc_factor_mat_solver_type"] = "mumps"

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A_mat)
    ksp.setOptionsPrefix(f"ksp_{id(ksp)}")

    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts.prefixPush(option_prefix)
    for option, value in fieldsplit_opts.items():
        opts[option] = value
    opts.prefixPop()

    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitIS(("A", is_A), ("V", is_V))

    ksp.setFromOptions()

    # Consistent right-hand side sol = A_mat x_rand (sol in range A_mat, cf. Remark 3.6)
    x_rand, sol = A_mat.createVecRight(), A_mat.createVecLeft()
    x_rand.setRandom()
    A_mat.mult(x_rand, sol)
    uh = A_mat.createVecRight()

    par_print(comm, f"--- {ksp_type.upper()} + additive fieldsplit, exact LU blocks ---")
    ksp.solve(sol, uh)

    res = A_mat * uh - sol
    par_print(comm, f"Relative residual norm: {res.norm() / sol.norm():.2e}")
    par_print(comm, f"Number of iterations: {ksp.getIterationNumber()}")
    par_print(comm, f"Convergence reason: {ksp.getConvergedReason()}")


run("cg")
run("gmres")
