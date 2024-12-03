import numpy as np
from mpi4py import MPI
import sys
from dolfinx.fem import assemble_scalar, form, functionspace
from ufl import dx, inner
from ufl.core.expr import Expr
from dolfinx import mesh

def par_print(comm, string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()

def L2_norm(v: Expr):
    """Computes the L2-norm of v
    """
    return np.sqrt(MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(inner(v, v) * dx)), op=MPI.SUM))

def boundary_marker(x):
    """Marker function for the boundary of a unit cube"""
    # Collect boundaries perpendicular to each coordinate axis
    boundaries = [
        np.logical_or(np.isclose(x[i], 0.0), np.isclose(x[i], 1.0))
        for i in range(3)]
    return np.logical_or(np.logical_or(boundaries[0],
                                        boundaries[1]),
                            boundaries[2])

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