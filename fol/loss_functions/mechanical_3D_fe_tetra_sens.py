"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2024
 License: FOL/License.txt
"""
from  .mechanical_3D_fe_tetra import MechanicalLoss3DTetra
import jax
import jax.numpy as jnp
from jax import jit,grad
from functools import partial
from fol.tools.decoration_functions import *
from fol.tools.fem_utilities import *
from fol.computational_models.fe_model import FiniteElementModel

class MechanicalLoss3DTetraSens(MechanicalLoss3DTetra):

    @partial(jit, static_argnums=(0,))
    def ComputeSingleLoss(self,known_dofs,unknown_dofs):
        full_UVW = self.GetFullDofVector(known_dofs,unknown_dofs)
        psudo_k = jnp.ones(int(full_UVW.shape[0]/3))
        residuals = self.ComputeResiduals(psudo_k,full_UVW)
        return residuals
 