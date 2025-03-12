# %%
import gmsh
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI

def box_with_inner(comm, h):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",0) # Supress output in terminal

    model = gmsh.model()

    name = "Box"
    vol_ids = {"inner": 1, "outer": 2}
    bound_ids = {"inner": 3, "outer": 4}

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
    boundary_ids = [b[1] for b in boundary]

    model.addPhysicalGroup(2, boundary_ids[:6], tag = bound_ids["inner"])  # Inner Box Boundary
    model.addPhysicalGroup(2, boundary_ids[6:12], tag = bound_ids["outer"] )  # Outer Box Boundary

    model.mesh.setSize(gmsh.model.getEntities(0), h)
    model.mesh.generate(dim=3)
    # gmsh.write("box.msh")

    model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    msh, ct, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank)

    ct.name = "ct"
    ft.name = "ft"

    gmsh.finalize()

    # with XDMFFile(msh.comm, "box_with_inner.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(msh)
    #     xdmf.write_meshtags(ct, msh.geometry)
    #     xdmf.write_meshtags(ft, msh.geometry)

    return msh, ct, ft, vol_ids, bound_ids

comm = MPI.COMM_WORLD
mesh, ct, ft, vol_ids, bound_ids = box_with_inner(comm, 0.1)


#%%

