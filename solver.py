from mpi4py import MPI
import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import mesh, fem, default_scalar_type, io
from dolfinx.fem import functionspace, Function, Expression, dirichletbc, form, assemble_scalar, locate_dofs_topological
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from ufl import SpatialCoordinate, variable, curl, grad, div, sin, cos, pi, diff, Measure
from basix.ufl import element
from ufl.core.expr import Expr


def solver(domain, domain_tags, degree, uex, uex1, d_t, ft, nu, sigma, bc_dict, f0, f1, t, num_steps, other):
    
    default_other = {
        "postpro": False,
        "output_prefix": "",
        "save_frequency": 1
    }
    
    # Update with user provided values
    if other is not None:
        default_other.update(other)
    other = default_other

    dx = Measure("dx", domain=domain, subdomain_data=domain_tags)

    nedelec_elem = element("N1curl", domain.basix_cell(), degree)
    V = functionspace(domain, nedelec_elem)
    lagrange_element = element("Lagrange", domain.basix_cell(), degree)
    V1 = functionspace(domain, lagrange_element)

    gdim = domain.geometry.dim
    facet_dim = gdim - 1

    dt = fem.Constant(domain, d_t)

    if uex != 0:
        print("Manufacturing solution")
        u_expr = Expression(uex, V.element.interpolation_points())
        u_expr1 = Expression(uex1, V1.element.interpolation_points())

        u_n = fem.Function(V)
        u_n.interpolate(u_expr)

        u_n1 = fem.Function(V1)
        u_n1.interpolate(u_expr1)
    else:
        u_n = fem.Function(V)
        u_n1 = fem.Function(V1)

    bc0_list = []
    bc1_list = []

    for space_type, conditions in bc_dict.items():
        if space_type == "V":
            for facet_tags, values in conditions.items():
                if isinstance(values, float):
                    value = fem.Constant(domain, (values, values, values))
                else:
                    value = values
                
                boundary_entities = np.concatenate([ft.find(tag) for tag in facet_tags])
                bdofs0 = locate_dofs_topological(V, facet_dim, boundary_entities)
                u_bc_V = Function(V)
                
                u_expr = Expression(value, V.element.interpolation_points())
                u_bc_V.interpolate(u_expr)
                bc0_list.append(dirichletbc(u_bc_V, bdofs0))
            
        elif space_type == "V1":
            for facet_tags, values in conditions.items():
                if isinstance(values, float):
                    value = fem.Constant(domain, values)
                else:
                    value = values
                
                boundary_entities = np.concatenate([ft.find(tag) for tag in facet_tags])
                bdofs1 = locate_dofs_topological(V1, facet_dim, boundary_entities)
                u_bc_V1 = Function(V1)
                
                u_expr1 = Expression(value, V1.element.interpolation_points())
                u_bc_V1.interpolate(u_expr1)
                bc1_list.append(dirichletbc(u_bc_V1, bdofs1))

    bc = bc0_list + bc1_list

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a00 = dt*nu*ufl.inner(curl(u), curl(v)) * dx + sigma*ufl.inner(u, v) * dx

    L0 = dt* ufl.inner(f0, v) * dx + sigma*ufl.inner(u_n, v) * dx

    u1 = ufl.TrialFunction(V1)
    v1 = ufl.TestFunction(V1)

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

    if other["postpro"]:
        vector_vis = fem.functionspace(domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,)))
        prefix = other["output_prefix"]
        
        if "A" in other["output_fields"]:
            A_vis = fem.Function(vector_vis)
            A_file = io.VTXWriter(domain.comm, f"{prefix}A.bp", A_vis, "BP4")
        if "B" in other["output_fields"]:
            B_vis = fem.Function(vector_vis)
            B_file = io.VTXWriter(domain.comm, f"{prefix}B.bp", B_vis, "BP4")
        if "V" in other["output_fields"]:
            V_file = io.VTXWriter(domain.comm, f"{prefix}V.bp", u_n1, "BP4")
        if "J" in other["output_fields"]:
            J_vis = fem.Function(vector_vis)
            J_file = io.VTXWriter(domain.comm, f"{prefix}J.bp", J_vis, "BP4")
    
    
    for n in range(num_steps):
        t.expression().value += dt.value
        
        if uex != 0:
            u_bc_V.interpolate(u_expr)
            u_bc_V1.interpolate(u_expr1)

        b = assemble_vector_block(L, a, bcs=bc)
        u_n_prev = u_n.copy()

        sol = A_mat.createVecRight()
        ksp.solve(b, sol)

        E = -(u_n - u_n_prev)/dt - ufl.grad(u_n1)
        J = sigma * E
        
        if other["postpro"] and n % other["save_frequency"] == 0:
            if "A" in other["output_fields"]:
                A_vis.interpolate(u_n)
                A_file.write(t.expression().value)
            
            if "B" in other["output_fields"]:
                B_3D = curl(u_n)
                Bexpr = fem.Expression(B_3D, vector_vis.element.interpolation_points())
                B_vis.interpolate(Bexpr)
                B_file.write(t.expression().value)
            
            if "V" in other["output_fields"]:
                V_file.write(t.expression().value)

            if "J" in other["output_fields"]:
                J_expr = Expression(J, vector_vis.element.interpolation_points())
                J_vis.interpolate(J_expr)
                J_file.write(t.expression().value)

        u_n.x.array[:] = sol.array[:offset]
        u_n1.x.array[:] = sol.array[offset:]

        u_n.x.scatter_forward()
        u_n1.x.scatter_forward()
    
    if other["postpro"] == True:
        if "A" in other["output_fields"]:
            A_file.close()
        if "B" in other["output_fields"]:
            B_file.close()
        if "V" in other["output_fields"]:
            V_file.close()
        if "J" in other["output_fields"]:
            J_file.close()

    return u_n, u_n1
