"""
 Authors: Kianoosh Taghikhani, https://github.com/kianoosh1989
 Date: July, 2024
 License: FOL/License.txt
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit,jacfwd,vmap
from jax.random import PRNGKey, normal
from scipy.spatial import KDTree
import numpy as np
from functools import partial
from jax.nn import sigmoid
from fol.tools.decoration_functions import *

class VoronoiControl(Control):
    @print_with_timestamp_and_execution_time
    def __init__(self,control_name: str,control_settings,fe_model):
        super().__init__(control_name)
        self.fe_model = fe_model
        self.settings = control_settings
        self.numberof_seeds = self.settings["numberof_seeds"]
        self.k_rangeof_values = self.settings["k_rangeof_values"]

        # The number 3 stands for the following: x coordinates array, y coordinates array, and K values
        self.num_control_vars = self.numberof_seeds * 3 
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
    def ComputeControlledVariables(self, variable_vector: jnp.array):
        x_coord = variable_vector[:self.numberof_seeds]
        y_coord = variable_vector[self.numberof_seeds:2 * self.numberof_seeds]
        k_values = variable_vector[2 * self.numberof_seeds:]
        X = np.asarray(self.fe_model.GetNodesX())
        Y = np.asarray(self.fe_model.GetNodesY())
        K = jnp.zeros((self.num_controlled_vars))
        seed_points = jnp.vstack((x_coord, y_coord)).T
        
        # Create the grid points
        grid_points = jnp.vstack([X.ravel(), Y.ravel()]).T
        
        # Calculate Euclidean distance between each grid point and each seed point
        def euclidean_distance(grid_point, seed_points):
            return jnp.sqrt(jnp.sum((grid_point - seed_points) ** 2, axis=1))
        
        # Iterate over grid points and assign the value from the nearest seed point
        def assign_value_to_grid(grid_point):
            distances = euclidean_distance(grid_point, seed_points)
            nearest_seed_idx = jnp.argmin(distances)
            return k_values[nearest_seed_idx]

        assign_value_to_grid_vmap = vmap(assign_value_to_grid)
        K = assign_value_to_grid_vmap(grid_points)

        return K

    @partial(jit, static_argnums=(0,))
    def ComputeJacobian(self,control_vec):
        return jnp.squeeze(jacfwd(self.ComputeControlledVariables,argnums=0)(control_vec))