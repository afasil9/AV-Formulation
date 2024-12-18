#%%
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from dolfinx import default_scalar_type
from dolfinx.fem.petsc import assemble_vector_block, assemble_matrix_block
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from basix.ufl import element
from ufl import (
    TrialFunction,
    TestFunction,
    inner,
    grad,
    div,
    curl,
    cross,
    dot,
    variable,
    as_vector,
    diff,
    sin,
    cos,
    pi,
    Measure,
    FacetNormal,
    SpatialCoordinate,
)
from dolfinx.fem import (
    dirichletbc,
    form,
    Function,
    Expression,
    locate_dofs_topological,
    functionspace,
    Constant,
)
from utils import L2_norm, create_mesh_fenics, par_print
import json
from dolfinx.common import Timer

comm = MPI.COMM_WORLD
degree = 1

n_values = [2,4]
results = {
    "dofs": [],
    "setup_times": [],
    "solve_times": [],
    "iterations": [],
    "E_errors": [],
    "B_errors": []
}

for n in n_values:

    timer_setup = Timer("Setup")
    timer_setup.start()

    ti = 0.0  # Start time
    T = 0.1  # End time
    num_steps = 100  # Number of time steps
    d_t = (T - ti) / num_steps  # Time step size

    boundaries = {"bottom": 1, "top": 2, "front": 3, "right": 4, "back": 5, "left": 6}
    domain, ft, ct = create_mesh_fenics(comm, n, boundaries)

    x = SpatialCoordinate(domain)
    t = variable(Constant(domain, ti))
    nu_ = 1.0
    sigma_ = 1.0

    def exact(x, t):
        return as_vector((
            x[1]**2 + x[0] * t,
            x[2]**2 + x[1] * t,
            x[0]**2 + x[2] * t))

    def exact1(x):
        return (x[0]**2) + (x[1]**2) + (x[2]**2)

    uex = exact(x, t)
    uex1 = exact1(x)

    f0 = nu_ * curl(curl(uex)) + sigma_ * diff(uex, t) + sigma_ * grad(uex1)
    f1 = -div(sigma_ * grad(uex1)) - div(sigma_ * diff(uex, t))

    preconditioner = "AMS"

    domain_tags = ct
    facet_tags = ft

    norm = FacetNormal(domain)
    hn0 = cross(nu_ * curl(uex), norm)
    hn1 = -dot(norm, sigma_ * diff(uex, t) + sigma_ * grad(uex1))

    bc_dict = {
        "dirichlet": {
            "V": {
                (
                    boundaries["bottom"],
                    boundaries["top"],
                    boundaries["back"],
                    boundaries["left"],
                ): uex
            },
            "V1": {
                (
                    boundaries["bottom"],
                    boundaries["top"],
                    boundaries["back"],
                    boundaries["left"],
                ): uex1
            },
        },
        "neumann": {
            "V": {(boundaries["front"], boundaries["right"]): uex},
            "V1": {(boundaries["front"], boundaries["right"]): uex1},
        },
    }


    # Solver

    dx = Measure("dx", domain=domain, subdomain_data=domain_tags)
    ds = Measure("ds", domain=domain, subdomain_data=facet_tags)
    dt = Constant(domain, d_t)
    nu = Constant(domain, default_scalar_type(nu_))
    sigma = Constant(domain, default_scalar_type(sigma_))

    gdim = domain.geometry.dim
    facet_dim = gdim - 1

    nedelec_elem = element("N1curl", domain.basix_cell(), degree)
    V = functionspace(domain, nedelec_elem)
    lagrange_elem = element("Lagrange", domain.basix_cell(), degree)
    V1 = functionspace(domain, lagrange_elem)

    u_n = Function(V)
    u_expr = Expression(uex, V.element.interpolation_points())
    u_n.interpolate(u_expr)

    u_n1 = Function(V1)
    uex_expr1 = Expression(uex1, V1.element.interpolation_points())
    u_n1.interpolate(uex_expr1)

    bc0_list = []
    bc1_list = []


    for category, conditions in bc_dict.items():
        for field, boundaries in conditions.items():
            if field == "V" and category == "dirichlet":
                if not boundaries:
                    par_print(comm, "No Dirichlet BCs present for Vector Potential")
                    is_dirichlet_V = False
                else:
                    par_print(comm, "Dirichlet BCs are present for Vector Potential")
                    is_dirichlet_V = True
                    for tags, value in boundaries.items():
                        boundary_entities = np.concatenate([ft.find(tag) for tag in tags])
                        bdofs = locate_dofs_topological(V, facet_dim, boundary_entities)
                        u_bc_V = Function(V)
                        u_expr_V = Expression(
                            value, V.element.interpolation_points(), comm=MPI.COMM_SELF
                        )
                        u_bc_V.interpolate(u_expr_V)
                        bc0_list.append(dirichletbc(u_bc_V, bdofs))
            elif field == "V1" and category == "dirichlet":
                if not boundaries:
                    par_print(comm,"No Dirichlet BCs present for Scalar Potential")
                    is_dirichlet_V1 = False
                else:
                    par_print(comm,"Dirichlet BCs are present for Scalar Potential")
                    is_dirichlet_V1 = True
                    for tags, value in boundaries.items():
                        boundary_entities = np.concatenate([ft.find(tag) for tag in tags])
                        bdofs = locate_dofs_topological(V1, facet_dim, boundary_entities)
                        u_bc_V1 = Function(V1)
                        u_expr_V1 = Expression(
                            value, V1.element.interpolation_points(), comm=MPI.COMM_SELF
                        )
                        u_bc_V1.interpolate(u_expr_V1)
                        bc1_list.append(dirichletbc(u_bc_V1, bdofs))
            elif field == "V" and category == "neumann":
                if not boundaries:
                    par_print(comm,"No Neumann BCs present for Vector Potential")
                    neumann_tags_V = None
                else:
                    par_print(comm,"Neumann BCs are present for Vector Potential")
                    for tags, value in boundaries.items():
                        neumann_tags_V = tags
            elif field == "V1" and category == "neumann":
                if not boundaries:
                    par_print(comm,"No Neumann BCs present for Scalar Potential")
                    neumann_tags_V1 = None
                else:
                    par_print(comm,"Neumann BCs are present for Scalar Potential")
                    for tags, value in boundaries.items():
                        neumann_tags_V1 = tags

    bc = bc0_list + bc1_list

    u = TrialFunction(V)
    v = TestFunction(V)

    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)

    u_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    u1_dofs = V1.dofmap.index_map.size_global * V1.dofmap.index_map_bs
    total_dofs = u_dofs + u1_dofs
    par_print(comm, f"Total degrees of freedom: {total_dofs}")

    #%%
    a00 = dt * nu * inner(curl(u), curl(v)) * dx + sigma * inner(u, v) * dx

    a01 = dt * sigma * inner(grad(u1), v) * dx
    a10 = sigma * inner(grad(v1), u) * dx

    a11 = dt * inner(sigma * grad(u1), grad(v1)) * dx

    if neumann_tags_V != None:
        L0 = (
            dt * inner(f0, v) * dx
            + sigma * inner(u_n, v) * dx
            + dt * inner(hn0, v) * ds(neumann_tags_V)
        )
    else:
        L0 = dt * inner(f0, v) * dx + sigma * inner(u_n, v) * dx

    if neumann_tags_V1 != None:
        L1 = (
            dt * f1 * v1 * dx
            + sigma * inner(grad(v1), u_n) * dx
            - dt * inner(hn1, v1) * ds(neumann_tags_V1)
        )
    else:
        L1 = dt * f1 * v1 * dx + sigma * inner(grad(v1), u_n) * dx


    a = form([[a00, a01], [a10, a11]])

    A_mat = assemble_matrix_block(a, bcs=bc)
    A_mat.assemble()

    L = form([L0, L1])
    b = assemble_vector_block(L, a, bcs=bc)

    if preconditioner == "Direct":
        par_print(comm,"Direct solve")
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
        par_print(comm,"AMS preconditioner")

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
            pc0.setHYPRESetEdgeConstantVectors(cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec)
        else:
            Vec_CG = functionspace(domain, ("CG", degree, (domain.geometry.dim,)))
            Pi = interpolation_matrix(Vec_CG._cpp_object, V._cpp_object)
            Pi.assemble()

            # Attach discrete gradient to preconditioner
            pc0.setHYPRESetInterpolations(domain.geometry.dim, None, None, Pi, None)

        opts = PETSc.Options()
        opts[f"{ksp_u.prefix}pc_hypre_ams_cycle_type"] = 14
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

#%%

    u_n_prev = u_n.copy()

    uh, uh1 = Function(V), Function(V1)
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    timer_setup.stop()
    setup_time = timer_setup.elapsed()[0]
    par_print(comm, f"Setup time: {setup_time} seconds")

    timer_solve = Timer("Solve")
    timer_solve.start()

    sol = A_mat.createVecRight()
    ksp.solve(b, sol)

    uh.x.array[:offset] = sol.array_r[:offset]
    uh1.x.array[:(len(sol.array_r) - offset)] = sol.array_r[offset:]

    uh.x.scatter_forward()
    uh1.x.scatter_forward()

    u_n.x.array[:] = uh.x.array
    u_n1.x.array[:] = uh1.x.array

    u_n.x.scatter_forward()
    u_n1.x.scatter_forward()

    vector_vis = functionspace(
        domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,))
    )

    da_dt = (u_n - u_n_prev) / dt
    E = -grad(u_n1) - da_dt
    B = curl(u_n)
    J = sigma * E

    for i in range(num_steps):
        t.expression().value += d_t
        # par_print(comm, f"Time step {n}")

        u_n_prev = u_n.copy()

        if is_dirichlet_V == True:
            u_bc_V.interpolate(u_expr_V)
        if is_dirichlet_V1 == True:
            u_bc_V1.interpolate(u_expr_V1)

        b = assemble_vector_block(L, a, bcs=bc)

        sol = A_mat.createVecRight()
        ksp.solve(b, sol)

        uh.x.array[:offset] = sol.array_r[:offset]
        uh1.x.array[:(len(sol.array_r) - offset)] = sol.array_r[offset:]

        uh.x.scatter_forward()
        uh1.x.scatter_forward()

        u_n.x.array[:] = uh.x.array
        u_n1.x.array[:] = uh1.x.array

        u_n.x.scatter_forward()
        u_n1.x.scatter_forward()

    iteration_count = ksp.getIterationNumber()
    # par_print(comm, iteration_count)

    timer_solve.stop()
    solve_time = timer_solve.elapsed()[0]
    par_print(comm, f"Total solve time: {solve_time} seconds")

    da_dt = (u_n - u_n_prev) / dt
    E = -grad(u_n1) - da_dt
    B = curl(u_n)

    # Post pro

    t_prev = T - d_t

    uex_prev = exact(x, t_prev)
    uex_final = exact(x, T)

    da_dt_exact = (uex_final - uex_prev) / d_t
    E_exact = -grad(uex1) - da_dt_exact

    E_error = L2_norm(E - E_exact)
    B_error = L2_norm(B - curl(uex))

    # par_print(comm, f"E field error {L2_norm(E - E_exact)}")
    # par_print(comm, f"B field error {L2_norm(B - curl(uex))}")

    results["dofs"].append(int(total_dofs))
    results["setup_times"].append(float(setup_time))
    results["solve_times"].append(float(solve_time))
    results["iterations"].append(int(iteration_count))
    results["E_errors"].append(float(E_error))
    results["B_errors"].append(float(B_error))

if comm.rank == 0:
    with open("convergence_results.json", "w") as f:
        json.dump(results, f, indent=4)
# %%