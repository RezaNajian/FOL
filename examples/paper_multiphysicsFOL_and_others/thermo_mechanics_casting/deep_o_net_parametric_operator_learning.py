"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2025
 License: FOL/LICENSE
"""

from typing import Iterator,Tuple 
import jax
import jax.numpy as jnp
from jax import jit,vmap
from functools import partial
from optax import GradientTransformation
from flax import nnx
from deep_network_multiphysics import DeepNetwork
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *

class DeepONetParametricOperatorLearning(DeepNetwork):

    def __init__(self,
                 name:str,
                 control:Control,
                 loss_function:Loss,
                 flax_neural_network:nnx.Module,
                 optax_optimizer:GradientTransformation):

        super().__init__(name,loss_function,flax_neural_network,
                         optax_optimizer)
        self.control = control
        
    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
 
        if self.initialized and not reinitialize:
            return

        super().Initialize(reinitialize)

        if not self.control.initialized:
            self.control.Initialize(reinitialize)

        self.initialized = True

    @print_with_timestamp_and_execution_time
    def Predict(self,batch_control:jnp.ndarray):
        batch_X = jax.vmap(self.control.ComputeControlledVariables)(batch_control)
        batch_Y =jax.vmap(self.flax_neural_network,(0,None))(batch_X,self.loss_function.fe_mesh.GetNodesCoordinates())
        batch_Y = batch_Y.reshape(batch_X.shape[0], -1)[:,self.loss_function.non_dirichlet_indices]
        return jax.vmap(self.loss_function.GetFullDofVector)(batch_X,batch_Y)

    def Finalize(self):
        pass

class DataDrivenDeepONetParametricOperatorLearning(DeepONetParametricOperatorLearning):

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,x_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        control_output = self.control.ComputeControlledVariables(x_set[0])
        nn_output = nn_model(control_output,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
        return self.loss_function.ComputeSingleLoss(x_set[1],nn_output)
    
class PhysicsInformedDeepONetParametricOperatorLearning(DeepONetParametricOperatorLearning):

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,x_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        control_output = self.control.ComputeControlledVariables(x_set[0])
        nn_output = nn_model(control_output,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
        return self.loss_function.ComputeSingleLoss(control_output.flatten(),nn_output)

class PhysicsInformedDeepONetParametricOperatorLearningDBC(DeepONetParametricOperatorLearning):

    def _ComputeThermoMechanicalLoss(self, control_output, nn_output):
        num_nodes = self.loss_function.fe_mesh.GetNumberOfNodes()
        num_dofs = self.loss_function.number_dofs_per_node
        full_dofs = nn_output.at[self.loss_function.dirichlet_indices].set(control_output)
        nodal_dofs = full_dofs.reshape(num_nodes, num_dofs)

        element_nodes = self.loss_function.fe_mesh.GetElementsNodes(
            self.loss_function.element_type
        )
        element_xyz = self.loss_function.fe_mesh.GetNodesCoordinates()[element_nodes, :]
        element_controls = jnp.ones(num_nodes)[element_nodes]
        element_dofs = nodal_dofs[element_nodes].reshape(element_nodes.shape[0], -1, 1)
        element_t0 = self.loss_function.thermal_loss_settings["T0"][element_nodes]

        element_thermal_losses, element_mechanical_losses = jax.vmap(
            lambda xyze, de, tuvwe, t0e: (
                self.loss_function.ComputeElementThermal(
                    xyze, de, tuvwe[self.loss_function.elem_t_local_ids]
                )[0],
                self.loss_function.ComputeElementMechanical(
                    xyze,
                    de,
                    tuvwe[self.loss_function.elem_t_local_ids],
                    tuvwe[self.loss_function.elem_uvw_local_ids],
                    t0e,
                )[0],
            )
        )(element_xyz, element_controls, element_dofs, element_t0)
        thermal_loss = jnp.sum(element_thermal_losses)
        mechanical_loss = jnp.sum(element_mechanical_losses)
        total_loss = (thermal_loss + mechanical_loss) \
            ** self.loss_function.loss_function_exponent
        return total_loss, (
            total_loss, total_loss, total_loss, thermal_loss, mechanical_loss
        )

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,x_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        control_output = self.control.ComputeControlledVariables(x_set[0])
        nn_output = nn_model(
            x_set[0], self.loss_function.fe_mesh.GetNodesCoordinates()
        ).flatten()
        return self._ComputeThermoMechanicalLoss(control_output, nn_output)

    @print_with_timestamp_and_execution_time
    def Predict(self,batch_control:jnp.ndarray):
        batch_X = jax.vmap(self.control.ComputeControlledVariables)(batch_control)
        batch_Y =jax.vmap(self.flax_neural_network,(0,None))(batch_control,self.loss_function.fe_mesh.GetNodesCoordinates())
        batch_Y = batch_Y.reshape(batch_X.shape[0], -1)
        return batch_Y.at[:, self.loss_function.dirichlet_indices].set(batch_X)
