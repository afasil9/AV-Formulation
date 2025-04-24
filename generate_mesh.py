# %%
import gmsh
from dolfinx.io import XDMFFile, gmshio
from dolfinx import mesh
from mpi4py import MPI

#%%

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

    msh = mesh_data[0]
    ct = mesh_data[1]
    ft = mesh_data[2]

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

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
        
    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=d, partitioner=partitioner)
    
    msh = mesh_data[0]
    ct = mesh_data[1]
    ft = mesh_data[2]
    
    gmsh.finalize()
    return msh, ct, ft, vol_ids, bound_ids



#%%

