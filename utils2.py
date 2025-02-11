import sys

import gmsh
import numpy as np
from dolfinx import io, mesh
from dolfinx.fem import assemble_scalar, form
from mpi4py import MPI
from ufl import dx, inner
from ufl.core.expr import Expr

def par_print(comm, string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()

def monitor(ksp, its, rnorm):
        iteration_count = []
        residual_norm = []
        iteration_count.append(its)
        residual_norm.append(rnorm)
        print("Iteration: {}, preconditioned residual: {}".format(its, rnorm))
