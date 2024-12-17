import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import default_scalar_type
from dolfinx.fem import dirichletbc,form, Function, Expression, locate_dofs_topological, functionspace, Constant
from dolfinx.fem.petsc import assemble_vector_block, assemble_matrix_block
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.io import VTXWriter
from basix.ufl import element
from ufl import grad, inner, curl, Measure, TrialFunction, TestFunction, variable, div

def solver_sym(comm, domain, facet_tags, domain_tags, degree, nu_, sigma_, t, d_t, num_steps, f0, f1, bc_dict, hn0, hn1, uex, wex, preconditioner, results):

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

    w_n = Function(V1)
    wex_expr = Expression(wex, V1.element.interpolation_points())
    w_n.interpolate(wex_expr)

    bc0_list = []
    bc1_list = []

    for category, conditions in bc_dict.items():
        for field, boundaries in conditions.items():
            if field == "V" and category == "dirichlet":
                if not boundaries:
                    print("No Dirichlet BCs present for Vector Potential")
                    is_dirichlet_V = False
                else:
                    print("Dirichlet BCs are present for Vector Potential")
                    is_dirichlet_V = True
                    for tags, value in boundaries.items():
                        boundary_entities = np.concatenate([facet_tags.find(tag) for tag in tags])
                        bdofs = locate_dofs_topological(V, facet_dim, boundary_entities)
                        u_bc_V = Function(V)
                        u_expr_V = Expression(
                            value, V.element.interpolation_points(), comm=MPI.COMM_SELF
                        )
                        u_bc_V.interpolate(u_expr_V)
                        bc0_list.append(dirichletbc(u_bc_V, bdofs))
            elif field == "V1" and category == "dirichlet":
                if not boundaries:
                    print("No Dirichlet BCs present for Scalar Potential")
                    is_dirichlet_V1 = False
                else:
                    print("Dirichlet BCs are present for Scalar Potential")
                    is_dirichlet_V1 = True
                    for tags, value in boundaries.items():
                        boundary_entities = np.concatenate([facet_tags.find(tag) for tag in tags])
                        bdofs = locate_dofs_topological(V1, facet_dim, boundary_entities)
                        w_bc_V1 = Function(V1)
                        w_expr_V1 = Expression(
                            value, V1.element.interpolation_points(), comm=MPI.COMM_SELF
                        )
                        w_bc_V1.interpolate(w_expr_V1)
                        bc1_list.append(dirichletbc(w_bc_V1, bdofs))
            elif field == "V" and category == "neumann":
                if not boundaries:
                    print("No Neumann BCs present for Vector Potential")
                    neumann_tags_V = None
                else:
                    print("Neumann BCs are present for Vector Potential")
                    for tags, value in boundaries.items():
                        neumann_tags_V = tags
            elif field == "V1" and category == "neumann":
                if not boundaries:
                    print("No Neumann BCs present for Scalar Potential")
                    neumann_tags_V1 = None
                else:
                    print("Neumann BCs are present for Scalar Potential")
                    for tags, value in boundaries.items():
                        neumann_tags_V1 = tags

    bc = bc0_list + bc1_list

    u = TrialFunction(V)
    v = TestFunction(V)

    w1 = TrialFunction(V1)
    v1 = TestFunction(V1)

    a00 = dt * nu * inner(curl(u), curl(v)) * dx + sigma * inner(u, v) * dx

    a01 = sigma * inner(grad(w1), v) * dx
    a10 = sigma * inner(grad(v1), u) * dx

    a11 = dt * inner(sigma * grad(w1), grad(v1)) * dx


    if neumann_tags_V != None:
        L0 = (
            dt * inner(f0, v) * dx
            + sigma * inner(u_n, v) * dx
            + sigma * inner(v, grad(w_n)) * dx
            + dt * inner(hn0, v) * ds(neumann_tags_V)
        )
    else:
        L0 = (
            dt * inner(f0, v) * dx
            + sigma * inner(u_n, v) * dx
            + sigma * inner(v, grad(w_n)) * dx
        )


    if neumann_tags_V1 != None:
        L1 = (
            dt * f1 * v1 * dx
            + sigma * inner(grad(v1), u_n) * dx
            + sigma * inner(grad(v1), grad(w_n)) * dx
            - dt * inner(hn1, v1) * ds(neumann_tags_V1)
        )
    else:
        L1 = (
            dt * f1 * v1 * dx
            + sigma * inner(grad(v1), u_n) * dx
            + sigma * inner(grad(v1), grad(w_n)) * dx
        )

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
            pc0.setHYPRESetEdgeConstantVectors(cvec_0.vector, cvec_1.vector, cvec_2.vector)
        else:
            Vec_CG = functionspace(domain, ("CG", degree, (domain.geometry.dim,)))
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
    w_n_prev = w_n.copy()

    uh, uh1 = Function(V), Function(V1)
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    sol = A_mat.createVecRight()
    ksp.solve(b, sol)

    uh.x.array[:] = sol.array_r[:offset]
    uh1.x.array[:] = sol.array_r[offset:]

    u_n.x.array[:] = uh.x.array
    w_n.x.array[:] = uh1.x.array

    print(ksp.getTolerances())

    vector_vis = functionspace(
        domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,))
    )

    da_dt = (u_n - u_n_prev) / dt
    dw_dt = (w_n - w_n_prev) / dt

    E = -grad(dw_dt) - da_dt
    B = curl(u_n)
    J = sigma * E

    if results["postpro"] == True:
        if "A" in results["output_fields"]:
            A_vis = Function(vector_vis)
            A_file = VTXWriter(domain.comm, "A.bp", A_vis, "BP4")
            A_vis.interpolate(u_n)
            A_file.write(t.expression().value)
        if "B" in results["output_fields"]:
            B_vis = Function(vector_vis)
            B_file = VTXWriter(domain.comm, "B.bp", B_vis, "BP4")
            Bexpr = Expression(B, vector_vis.element.interpolation_points())
            B_vis.interpolate(Bexpr)
            B_file.write(t.expression().value)
        if "V" in results["output_fields"]:
            V_file = VTXWriter(domain.comm, "V.bp", w_n, "BP4")
            V_file.write(t.expression().value)
        if "J" in results["output_fields"]:
            J_vis = Function(vector_vis)
            J_expr = Expression(J, vector_vis.element.interpolation_points())
            J_vis.interpolate(J_expr)
            J_file = VTXWriter(domain.comm, "J.bp", J_vis, "BP4")
        if "E" in results["output_fields"]:
            E_vis = Function(vector_vis)
            E_expr = Expression(E, vector_vis.element.interpolation_points())
            E_vis.interpolate(E_expr)
            E_file = VTXWriter(domain.comm, "E.bp", E_vis, "BP4")


    for n in range(num_steps):
        t.expression().value += d_t

        u_n_prev = u_n.copy()
        w_n_prev = w_n.copy()

        if is_dirichlet_V == True:
            u_bc_V.interpolate(u_expr_V)
        if is_dirichlet_V1 == True:
            w_bc_V1.interpolate(w_expr_V1)

        b = assemble_vector_block(L, a, bcs=bc)

        sol = A_mat.createVecRight()
        ksp.solve(b, sol)

        uh.x.array[:] = sol.array_r[:offset]
        uh1.x.array[:] = sol.array_r[offset:]

        u_n.x.array[:] = uh.x.array
        w_n.x.array[:] = uh1.x.array

        u_n.x.scatter_forward()
        w_n.x.scatter_forward()

        if results["postpro"] == True and n % results["save_frequency"] == 0:
            A_vis.interpolate(u_n)
            A_file.write(t.expression().value)

            V_file.write(t.expression().value)

            B = curl(u_n)
            B_vis.interpolate(Bexpr)
            B_file.write(t.expression().value)

            da_dt = (u_n - u_n_prev) / dt
            dw_dt = (w_n - w_n_prev) / dt

            E = -grad(dw_dt) - da_dt
            E_expr = Expression(E, vector_vis.element.interpolation_points())
            E_vis.interpolate(E_expr)
            E_file.write(t.expression().value)

            J = sigma * E
            J_expr = Expression(J, vector_vis.element.interpolation_points())
            J_vis.interpolate(J_expr)
            J_file.write(t.expression().value)

    if results["postpro"] == True:
        if "A" in results["output_fields"]:
            A_file.close()
        if "B" in results["output_fields"]:
            B_file.close()
        if "V" in results["output_fields"]:
            V_file.close()
        if "J" in results["output_fields"]:
            J_file.close()

    da_dt = (u_n - u_n_prev) / dt
    dw_dt = (w_n - w_n_prev) / dt

    E = -grad(dw_dt) - da_dt
    B = curl(u_n)
    J = sigma * E

    return E, B, J

