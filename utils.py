import numpy as np
from mpi4py import MPI
import sys
from dolfinx.fem import assemble_scalar, form, functionspace
from ufl import dx, inner
from ufl.core.expr import Expr

def par_print(comm, string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()

def L2_norm(v: Expr):
    """Computes the L2-norm of v
    """
    return np.sqrt(MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(inner(v, v) * dx)), op=MPI.SUM))