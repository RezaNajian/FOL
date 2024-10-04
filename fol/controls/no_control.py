"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit,jacfwd
from functools import partial
from jax.nn import sigmoid
from fol.tools.decoration_functions import *

class NoControl(Control):
    @print_with_timestamp_and_execution_time
    def __init__(self,control_name: str,fe_model):
        super().__init__(control_name)
        self.fe_model = fe_model
        self.num_control_vars = self.fe_model.GetNumberOfNodes()
        self.num_controlled_vars = self.fe_model.GetNumberOfNodes()

    def GetNumberOfVariables(self):
        return self.num_control_vars
    
    def GetNumberOfControlledVariables(self):
        return self.num_controlled_vars

    def Initialize(self) -> None:
        pass

    def Finalize(self) -> None:
        pass

    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self,variable_vector:jnp.array):
        return variable_vector
    
    @partial(jit, static_argnums=(0,))
    def ComputeJacobian(self,control_vec):
        pass