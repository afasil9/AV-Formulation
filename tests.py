#%% 
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import mesh, fem, default_scalar_type, io
from dolfinx.fem import functionspace, Function, Expression, dirichletbc, form, assemble_scalar, locate_dofs_topological
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from ufl import SpatialCoordinate, variable, curl, dx, grad, div, sin, cos, pi, diff, Measure
from basix.ufl import element
from ufl.core.expr import Expr
from dolfinx.io import XDMFFile, VTXWriter
import gmsh
from scipy.linalg import norm
from solver_draft1 import solver

problem = 1

def L2_norm(v: Expr):
    """Computes the L2-norm of v"""
    return np.sqrt(MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(ufl.inner(v, v) * ufl.dx)), op=MPI.SUM))

def create_mesh_gmsh(comm, n, boundaries):
    h = 1 / n
    gmsh.initialize()
    if comm.rank == 0:
        gmsh.model.add("model")
        factory = gmsh.model.geo

        # Create points for a unit cube
        points = [
            factory.addPoint(0.0, 0.0, 0.0, h),  # p0
            factory.addPoint(1.0, 0.0, 0.0, h),  # p1
            factory.addPoint(1.0, 1.0, 0.0, h),  # p2
            factory.addPoint(0.0, 1.0, 0.0, h),  # p3
            factory.addPoint(0.0, 0.0, 1.0, h),  # p4
            factory.addPoint(1.0, 0.0, 1.0, h),  # p5
            factory.addPoint(1.0, 1.0, 1.0, h),  # p6
            factory.addPoint(0.0, 1.0, 1.0, h),  # p7
        ]

        # Create lines connecting points
        lines = [
            # Bottom face
            factory.addLine(points[0], points[1]),  # l0
            factory.addLine(points[1], points[2]),  # l1
            factory.addLine(points[2], points[3]),  # l2
            factory.addLine(points[3], points[0]),  # l3
            # Top face
            factory.addLine(points[4], points[5]),  # l4
            factory.addLine(points[5], points[6]),  # l5
            factory.addLine(points[6], points[7]),  # l6
            factory.addLine(points[7], points[4]),  # l7
            # Vertical edges
            factory.addLine(points[0], points[4]),  # l8
            factory.addLine(points[1], points[5]),  # l9
            factory.addLine(points[2], points[6]),  # l10
            factory.addLine(points[3], points[7]),  # l11
        ]

        # Create curve loops for each face
        curve_loops = [
            factory.addCurveLoop([lines[0], lines[1], lines[2], lines[3]]),    # bottom
            factory.addCurveLoop([lines[4], lines[5], lines[6], lines[7]]),    # top
            factory.addCurveLoop([lines[0], lines[9], -lines[4], -lines[8]]),  # front
            factory.addCurveLoop([lines[1], lines[10], -lines[5], -lines[9]]), # right
            factory.addCurveLoop([lines[2], lines[11], -lines[6], -lines[10]]),# back
            factory.addCurveLoop([lines[3], lines[8], -lines[7], -lines[11]]), # left
        ]

        # Create surfaces for each face
        surfaces = [factory.addPlaneSurface([loop]) for loop in curve_loops]

        # Create surface loop and volume
        surface_loop = factory.addSurfaceLoop(surfaces)
        volume = factory.addVolume([surface_loop])

        factory.synchronize()

        # Add physical groups
        gmsh.model.addPhysicalGroup(3, [volume], 1)  # Volume
        
        # Add boundary markers
        gmsh.model.addPhysicalGroup(2, [surfaces[0]], boundaries["bottom"])
        gmsh.model.addPhysicalGroup(2, [surfaces[1]], boundaries["top"])
        gmsh.model.addPhysicalGroup(2, [surfaces[2]], boundaries["front"])
        gmsh.model.addPhysicalGroup(2, [surfaces[3]], boundaries["right"])
        gmsh.model.addPhysicalGroup(2, [surfaces[4]], boundaries["back"])
        gmsh.model.addPhysicalGroup(2, [surfaces[5]], boundaries["left"])

        gmsh.model.mesh.generate(3)
        # gmsh.fltk.run()  # Uncomment to visualize mesh

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
    msh, ct, ft = io.gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=3, partitioner=partitioner
    )
    gmsh.finalize()
    return msh, ft, ct

def exact(x, t):
    return ufl.as_vector((cos(pi * x[1]) * sin(pi * t), cos(pi * x[2]) * sin(pi * t), cos(pi * x[0]) * sin(pi * t)))

def exact1(x):
    return sin(pi*x[0]) * sin(pi*x[1]) * sin(pi*x[2])

def markers_to_meshtags(msh, tags, markers, dim):
    entities = [mesh.locate_entities_boundary(msh, dim, marker) for marker in markers]
    values = [np.full_like(entities, tag) for (tag, entities) in zip(tags, entities)]
    entities = np.hstack(entities, dtype=np.int32)
    values = np.hstack(values, dtype=np.intc)
    perm = np.argsort(entities)
    return mesh.meshtags(msh, dim, entities[perm], values[perm])

def create_mesh_fenics(comm, n, boundaries):
    # Create mesh
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, mesh.CellType.tetrahedron)

    # Create facet meshtags
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Define markers for all 6 faces of the cube
    markers = [
        lambda x: np.isclose(x[2], 0.0),  # bottom (z = 0)
        lambda x: np.isclose(x[2], 1.0),  # top (z = 1)
        lambda x: np.isclose(x[1], 0.0),  # front (y = 0)
        lambda x: np.isclose(x[0], 1.0),  # right (x = 1)
        lambda x: np.isclose(x[1], 1.0),  # back (y = 1)
        lambda x: np.isclose(x[0], 0.0),  # left (x = 0)
    ]
    
    # Create facet tags
    ft = markers_to_meshtags(msh, boundaries.values(), markers, fdim)
    
    # Create domain tags
    ct = mesh.meshtags(msh, tdim, np.array(range(msh.topology.index_map(tdim).size_local), dtype=np.int32), 
                      np.full(msh.topology.index_map(tdim).size_local, 1, dtype=np.int32))

    print("number of cells is", msh.topology.index_map(tdim).size_local)
    return msh, ft, ct

if problem == 1:
    ti = 0.0  # Start time
    T = 0.1  # End time
    num_steps = 400  # Number of time steps
    d_t = (T - ti) / num_steps  # Time step size

    n = 8
    degree = 1

    boundaries = {"bottom": 1, "top": 2, "front": 3, "right": 4, "back": 5, "left": 6}
    domain, ft, ct = create_mesh_gmsh(MPI.COMM_WORLD, n, boundaries)

    t = variable(fem.Constant(domain, ti))
    x = SpatialCoordinate(domain)
    uex = exact(x, t)   
    uex1 = exact1(x)

    dt = fem.Constant(domain, d_t)
    nu = fem.Constant(domain, default_scalar_type(10))
    sigma = fem.Constant(domain, default_scalar_type(10))

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

    f0 = nu*curl(curl(uex)) + sigma*diff(uex,t) + sigma*grad(uex1)
    f1 = -div(sigma*grad(uex1)) - div(sigma*diff(uex,t))

    u_n, u_n1 = solver(domain,ct, degree, uex, uex1, d_t, ft, nu, sigma, bc_dict, f0, f1, t, num_steps, other=None)

    print(L2_norm(curl(u_n - uex)))
    print(L2_norm(u_n1 - uex1))


if problem == 2:

    comm = MPI.COMM_WORLD
    boundaries = {"bottom": 1, "top": 2, "front": 3, "right": 4, "back": 5, "left": 6}
    n = 4
    domain, ft, ct = create_mesh_fenics(comm, n, boundaries)

    ti = 0.0  # Start time
    T = 0.1  # End time
    num_steps = 400  # Number of time steps
    d_t = (T - ti) / num_steps  # Time step size

    degree = 1
    t = variable(fem.Constant(domain, ti))
    x = SpatialCoordinate(domain)
    uex = exact(x, t)
    uex1 = exact1(x)

    dt = fem.Constant(domain, d_t)
    nu = fem.Constant(domain, default_scalar_type(10))
    sigma = fem.Constant(domain, default_scalar_type(10))

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

    f0 = nu*curl(curl(uex)) + sigma*diff(uex,t) + sigma*grad(uex1)
    f1 = -div(sigma*grad(uex1)) - div(sigma*diff(uex,t))

    u_n, u_n1 = solver(domain, ct, degree, uex, uex1, d_t, ft, nu, sigma, bc_dict, f0, f1, t, num_steps, other = None)

    print(L2_norm(curl(u_n - uex)))
    print(L2_norm(u_n1 - uex1))
