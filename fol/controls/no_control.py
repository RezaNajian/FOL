"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.mesh_input_output.mesh import Mesh
from fol.tools.decoration_functions import *

class NoControl(Control):
    def __init__(self,control_name: str, fe_mesh: Mesh):
        super().__init__(control_name)
        self.fe_mesh = fe_mesh

    def GetNumberOfVariables(self):
        return self.num_control_vars
    
    def GetNumberOfControlledVariables(self):
        return self.num_controlled_vars

    def Initialize(self) -> None:
        self.num_control_vars = self.fe_mesh.GetNumberOfNodes()
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()
        self.__initialized = True

    def Finalize(self) -> None:
        pass

    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self,variable_vector:jnp.array):
        return variable_vector
    
    @partial(jit, static_argnums=(0,))
    def ComputeJacobian(self,control_vec):
        pass