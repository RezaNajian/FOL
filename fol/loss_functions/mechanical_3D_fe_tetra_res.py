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

class MechanicalLoss3DTetraRes(MechanicalLoss3DTetra):

    @partial(jit, static_argnums=(0,))
    def ComputeSingleLoss(self,known_dofs,unknown_dofs):
        full_UVW = self.GetFullDofVector(known_dofs,unknown_dofs)
        psudo_k = jnp.ones(int(full_UVW.shape[0]/3))
        residuals = self.ComputeResiduals(psudo_k,full_UVW)
        residuals = residuals**2
        # min_res = jax.lax.stop_gradient(jnp.min(residuals))
        # max_res = jax.lax.stop_gradient(jnp.max(residuals))
        # residuals = (residuals-min_res)/(max_res-min_res)
        function_to_be_diff = jnp.mean(residuals)
        return function_to_be_diff,(0,0,function_to_be_diff)
 