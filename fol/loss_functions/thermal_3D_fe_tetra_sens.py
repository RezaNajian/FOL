"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2024
 License: FOL/License.txt
"""
from  .thermal_3D_fe_tetra_en import ThermalLoss3DTetraEnergy
import jax
import jax.numpy as jnp
from jax import jit,grad
from functools import partial
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.computational_models.fe_model import FiniteElementModel

class ThermalLoss3DTetraSens(ThermalLoss3DTetraEnergy):

    @partial(jit, static_argnums=(0,))
    def ComputeSingleLoss(self,full_control_params,unknown_dofs):
        full_dofs = self.ExtendUnknowDOFsWithBC(unknown_dofs)
        residuals = self.ComputeResiduals(full_control_params,full_dofs)
        return residuals
 