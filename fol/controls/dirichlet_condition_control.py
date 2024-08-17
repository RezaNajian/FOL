"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: August, 2024
 License: FOL/License.txt
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit,jacfwd
from functools import partial
from fol.computational_models.model import Model
from fol.tools.decoration_functions import *

class DirichletConditionControl(Control):
    @print_with_timestamp_and_execution_time
    def __init__(self,control_name:str,control_settings:dict,bc_settings:dict,comp_model:Model):
        super().__init__(control_name)
        self.comp_model = comp_model
        self.dof_settings = control_settings
        self.bc_settings = bc_settings
        if not self.dof_settings.keys() <= self.bc_settings.keys():
            error_msg = f"provided DoFs:{list(self.dof_settings.keys())} do not match the computational model's DoFs {list(self.bc_settings.keys())}"
            fol_error(error_msg)
        
        self.num_control_vars = 0
        for dof,boundary_list in self.dof_settings.items():
            for boundary in boundary_list:
                if not boundary in self.bc_settings[dof].keys():
                    error_msg = f"boundary {boundary} does not exist in dof {dof} settings of the model's bc"
                    fol_error(error_msg)
            
            self.num_control_vars += len(boundary_list)

    def GetNumberOfVariables(self):
        return self.num_control_vars
    
    def GetNumberOfControlledVariables(self):
        fol_error("this function is not implemented !")

    def Initialize(self) -> None:
        pass

    def Finalize(self) -> None:
        pass

    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self,variable_vector:jnp.array):
        if variable_vector.shape[-1] != self.num_control_vars:
            raise ValueError('shape of given coefficients does not match the number of frequencies !')
        
        kkkk
        
        # K = jnp.zeros((self.num_controlled_vars))
        # K += variable_vector[0]/2.0
        # coeff_counter = 1
        # for freq_x in self.x_freqs:
        #     for freq_y in self.y_freqs:
        #         for freq_z in self.z_freqs:
        #             K += variable_vector[coeff_counter] * jnp.cos(freq_x * jnp.pi * self.fe_model.GetNodesX()) * jnp.cos(freq_y * jnp.pi * self.fe_model.GetNodesY()) * jnp.cos(freq_z * jnp.pi * self.fe_model.GetNodesZ())
        #             coeff_counter += 1

        # return (self.max-self.min) * sigmoid(self.beta*(K-0.5)) + self.min
    
    @partial(jit, static_argnums=(0,))
    def ComputeJacobian(self,control_vec):
        return jnp.squeeze(jacfwd(self.ComputeControlledVariables,argnums=0)(control_vec))