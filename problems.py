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
from dolfinx.io import XDMFFile, VTXWriter
from solver_draft1 import solver

# Induction coil problem

comm = MPI.COMM_WORLD

with XDMFFile(MPI.COMM_WORLD, "em_model2_refined.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="domains")
    domain_tags = xdmf.read_meshtags(domain, "domains")
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    domain.topology.create_connectivity(1, tdim)
    ft = xdmf.read_meshtags(domain, "facets")


ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 1000  # Number of time steps
# d_t = (T - ti) / num_steps  # Time step size
d_t = 1e-4

degree = 1
t = variable(fem.Constant(domain, ti))
x = SpatialCoordinate(domain)
uex = 0
uex1 = 0

dt = fem.Constant(domain, d_t)

const = fem.functionspace(domain, ("DG", 0)) #Piecewise constant function space
mu_0 = 1.2566e-09 
mu_r = fem.Constant(domain, default_scalar_type(1.0))
mu_c = fem.Constant(domain, default_scalar_type(1.0))

sigma_air = fem.Constant(domain, default_scalar_type(1))
sigma_copper = fem.Constant(domain, default_scalar_type(5.96e4))

sigma = fem.Function(const)
mu = fem.Function(const)
nu = fem.Function(const)

def interpolate_by_tags(function, value_dict, domain_tags):
    for tag, value in value_dict.items():
        function.interpolate(lambda x: np.full_like(x[0], value), domain_tags.find(tag))
    function.x.scatter_forward()

sigma_values = {
    1: sigma_air,
    2: sigma_air,
    3: sigma_copper,
    4: sigma_air
}

mu_values = {
    1: mu_0 * mu_r,
    2: mu_0 * mu_r,
    3: mu_0 * mu_c,
    4: mu_0 * mu_r
}

nu_values = {
    1: 1 / (mu_0 * mu_r),
    2: 1 / (mu_0 * mu_r),
    3: 1 / (mu_0 * mu_c),
    4: 1 / (mu_0 * mu_r)
}

interpolate_by_tags(sigma, sigma_values, domain_tags)
interpolate_by_tags(nu, nu_values, domain_tags)

bc_dict = {
    'V': {
        (12, 13, 14, 15, 16, 18): 0.0,  # zeroA boundary condition
    },
    'V1': {
        (1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,18): 0.0,   # Note facet 9 is a sink facet
        (10,): 1.0,  # source (V_high)
    }
}

f0 = ufl.as_vector([0,0,0])
f1 = fem.Constant(domain, PETSc.ScalarType(0))

other = {
    "postpro": True,
    "output_prefix": "results_",
    "save_frequency": 100,
    "output_fields": ["A", "B", "V", "J"]
}

solver(domain, domain_tags, degree, uex, uex1, d_t, ft, nu, sigma, bc_dict, f0, f1, t, num_steps, other)

