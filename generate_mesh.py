import gmsh
from dolfinx.io import XDMFFile, gmsh as gmshio
from dolfinx import mesh
from mpi4py import MPI
import numpy as np

def box_with_inner(comm, h):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",0) # Supress output in terminal

    model = gmsh.model()

    name = "Box"
    vol_ids = {"inner": 0, "outer": 1}
    bound_ids = {"interface": 2, "boundary": 3}

    model.add(name)
    model.setCurrent(name)

    bbl = 1  # Big Box Length
    sc = 0.3  # Centre of the smaller box
    sbl = ((bbl / 2) - sc) * 2  # Small Box Length

    big_box = model.occ.addBox(0, 0, 0, bbl, bbl, bbl, 0)  # This tag includes the small box
    small_box = model.occ.addBox(sc, sc, sc, sbl, sbl, sbl, 1)

    model_dim_tags = model.occ.fragment(
        [(3, big_box)], [(3, small_box)]
    )  # This gives new dimension tags. Only interested in the first value of the tuple i.e. [0] which contains (dim, tag)
    model.occ.synchronize()

    model.addPhysicalGroup(3, [model_dim_tags[0][0][1]], tag = vol_ids["inner"])  # Inner Box tag
    model.addPhysicalGroup(3, [model_dim_tags[0][1][1]], tag = vol_ids["outer"])  # Outer Box Tag

    boundary = model.getBoundary([model_dim_tags[0][1]], oriented=False)
    interfaceds = [b[1] for b in boundary]

    model.addPhysicalGroup(2, interfaceds[:6], tag = bound_ids["interface"])  # Inner Box Boundary
    model.addPhysicalGroup(2, interfaceds[6:12], tag = bound_ids["boundary"] )  # Outer Box Boundary

    model.mesh.setSize(gmsh.model.getEntities(0), h)
    model.mesh.generate(dim=3)
    # gmsh.write("box.msh")

    model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank)

    msh = mesh_data.mesh
    ct = mesh_data.cell_tags
    ft = mesh_data.facet_tags

    ct.name = "ct"
    ft.name = "ft"

    gmsh.finalize()

    # with XDMFFile(msh.comm, "box_with_inner.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(msh)
    #     xdmf.write_meshtags(ct, msh.geometry)
    #     xdmf.write_meshtags(ft, msh.geometry)

    return msh, ct, ft, vol_ids, bound_ids

def create_box_with_sphere_msh(comm, h):
    """
    Create a mesh of a box containing a sphere.

    Parameters:
        h: maximum cell diameter
    """

    gmsh.initialize()
    d = 3  # Geometric dimension
    gmsh.option.setNumber("General.Terminal",0) # Supress output in terminal


    # Tags for volumes and boundaries
    vol_ids = {"inner": 0, "outer": 1}
    bound_ids = {
        "boundary": 2,  # Boundary
        "interface": 3,  # Interface
    }

    if comm.rank == 0:
        gmsh.model.add("box_with_sphere")
        box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        sphere = gmsh.model.occ.addSphere(0.5, 0.5, 0.5, 0.25)
        ov, ovv = gmsh.model.occ.fragment([(3, box)], [(3, sphere)])

        gmsh.model.occ.synchronize()

        ov = sorted(ov, key=lambda e: gmsh.model.occ.getMass(*e))

        # Add physical groups
        gmsh.model.addPhysicalGroup(3, [ov[0][1]], vol_ids["inner"])
        gmsh.model.addPhysicalGroup(3, [ov[1][1]], vol_ids["outer"])
        boundary_dim_tags = gmsh.model.getBoundary([ov[0], ov[1]])
        interface_dim_tags = gmsh.model.getBoundary([ov[0]])
        gmsh.model.addPhysicalGroup(
            2, [surface[1] for surface in boundary_dim_tags], bound_ids["boundary"]
        )
        gmsh.model.addPhysicalGroup(
            2, [surface[1] for surface in interface_dim_tags], bound_ids["interface"]
        )

        # Assign a mesh size to all the points:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

        gmsh.model.mesh.generate(d)

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none, 2)
        
    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=d, partitioner=partitioner)
    
    msh = mesh_data.mesh
    ct = mesh_data.cell_tags
    ft = mesh_data.facet_tags
    
    gmsh.finalize()
    return msh, ct, ft, vol_ids, bound_ids




def create_mesh_fenics(comm, n, size_inner=4.0, split_faces=False):
    """
    Create a hexahedral unit cube mesh with an inner conductive box centred
    at (0.5, 0.5, 0.5).

    Parameters:
        n: number of cells in each direction
        size_inner: half-width of the inner box in number of cells
        split_faces: tag each face of the inner box separately instead of
            using a single interface tag
    """

    msh = mesh.create_unit_cube(comm, n, n, n, cell_type=mesh.CellType.hexahedron)

    tdim = msh.topology.dim
    fdim = tdim - 1

    # Tags for volumes and boundaries
    vol_ids = {"inner": 1, "outer": 0}
    if split_faces:
        bound_ids = {
            "boundary": 2,
            "interface_xmin": 3,
            "interface_xmax": 4,
            "interface_ymin": 5,
            "interface_ymax": 6,
            "interface_zmin": 7,
            "interface_zmax": 8,
        }
    else:
        bound_ids = {
            "boundary": 2,  # Boundary
            "interface": 3,  # Interface
        }

    # Cells whose midpoint lies inside the inner box (including ghosts)
    cell_imap = msh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)
    mids = mesh.compute_midpoints(msh, tdim, cells)

    inner_loc = size_inner / n
    eps = 1e-12
    is_inner = np.all(np.abs(mids - 0.5) < inner_loc + eps, axis=1)

    cell_values = np.where(is_inner, vol_ids["inner"], vol_ids["outer"]).astype(np.int32)
    ct = mesh.meshtags(msh, tdim, cells, cell_values)

    # Outer boundary of the unit cube
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)

    # Interface = facets touching both regions
    inner_facets = mesh.compute_incident_entities(msh.topology, cells[is_inner], tdim, fdim)
    outer_facets = mesh.compute_incident_entities(msh.topology, cells[~is_inner], tdim, fdim)
    interface_facets = np.intersect1d(inner_facets, outer_facets)

    if split_faces:
        # Classify each interface facet by the plane of the inner box it lies in
        f_mids = mesh.compute_midpoints(msh, fdim, interface_facets.astype(np.int32))
        box_min = 0.5 - inner_loc
        box_max = 0.5 + inner_loc
        tol = 1e-6
        interface_values = np.zeros(interface_facets.size, dtype=np.int32)
        for i, ax in enumerate("xyz"):
            interface_values[np.abs(f_mids[:, i] - box_min) < tol] = bound_ids[f"interface_{ax}min"]
            interface_values[np.abs(f_mids[:, i] - box_max) < tol] = bound_ids[f"interface_{ax}max"]
    else:
        interface_values = np.full(interface_facets.size, bound_ids["interface"], dtype=np.int32)

    facets = np.concatenate([boundary_facets, interface_facets]).astype(np.int32)
    values = np.concatenate(
        [np.full(boundary_facets.size, bound_ids["boundary"], dtype=np.int32), interface_values]
    )
    order = np.argsort(facets)
    ft = mesh.meshtags(msh, fdim, facets[order], values[order])

    ct.name = "ct"
    ft.name = "ft"

    return msh, ct, ft, vol_ids, bound_ids
