"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: August, 2024
 License: FOL/License.txt
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit,jacfwd
from functools import partial
from fol.loss_functions.fe_loss import FiniteElementLoss
from fol.tools.decoration_functions import *

class DirichletConditionControl(Control):
    @print_with_timestamp_and_execution_time
    def __init__(self,control_name:str,control_settings:dict,fe_loss:FiniteElementLoss):
        super().__init__(control_name)
        self.control_settings = control_settings
        self.fe_loss = fe_loss
        self.fe_loss_dof_dict = self.fe_loss.GetLossDofsDict()
        self.dirichlet_dofs_boundary_dict = self.fe_loss_dof_dict["dirichlet_dofs_boundary_dict"]
        if not self.control_settings.keys() <= self.dirichlet_dofs_boundary_dict.keys():
            error_msg = f"provided DoFs:{list(self.control_settings.keys())} do not match the DoFs of provided loss function {self.fe_loss.GetName(),list(self.dirichlet_dofs_boundary_dict.keys())}"
            fol_error(error_msg)
        
        self.num_control_vars = 0
        for dof,boundary_list in self.control_settings.items():
            for boundary in boundary_list:
                if not boundary in self.dirichlet_dofs_boundary_dict[dof].keys():
                    error_msg = f"boundary {boundary} does not exist in dof {dof} settings of the loss's bc"
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
            fol_error('number of the input control variables does not match the number of control variables !')

        dirichlet_values = self.fe_loss_dof_dict["dirichlet_values"].copy()
        control_var_index = 0
        for dof,boundary_list in self.control_settings.items():
            for boundary in boundary_list:
                control_bc_value = variable_vector[control_var_index]
                dirichlet_bc_indices = self.dirichlet_dofs_boundary_dict[dof][boundary]
                control_bc_vector_value = control_bc_value * jnp.ones(dirichlet_bc_indices.size)
                dirichlet_values = dirichlet_values.at[dirichlet_bc_indices].set(control_bc_vector_value)
                control_var_index += 1

        return dirichlet_values
    
    @partial(jit, static_argnums=(0,))
    def ComputeJacobian(self,control_vec):
        return jnp.squeeze(jacfwd(self.ComputeControlledVariables,argnums=0)(control_vec))