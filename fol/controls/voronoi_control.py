"""
 Authors: Kianoosh Taghikhani, https://github.com/kianoosh1989
 Date: July, 2024
 License: FOL/License.txt
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit,vmap
from scipy.spatial import Voronoi, KDTree
import numpy as np
from functools import partial
from jax.nn import sigmoid
from fol.mesh_input_output.mesh import Mesh
from fol.tools.decoration_functions import *

class VoronoiControl(Control):

    def __init__(self,control_name: str,control_settings:dict, fe_mesh: Mesh):
        super().__init__(control_name)
        self.settings = control_settings
        self.fe_mesh = fe_mesh

    @print_with_timestamp_and_execution_time
    def Initialize(self) -> None:
        self.numberof_seeds = self.settings["numberof_seeds"]
        self.k_rangeof_values = self.settings["k_rangeof_values"]

        # The number 3 stands for the following: x coordinates array, y coordinates array, and K values
        self.num_control_vars = self.numberof_seeds * 3 
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()
        self.initialized = True

    def Finalize(self) -> None:
        pass
    
    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self, variable_vector: jnp.array):
        x_coord = variable_vector[:self.numberof_seeds]
        y_coord = variable_vector[self.numberof_seeds:2 * self.numberof_seeds]
        k_values = variable_vector[2 * self.numberof_seeds:]
        X = jnp.asarray(self.fe_mesh.GetNodesX())
        Y = jnp.asarray(self.fe_mesh.GetNodesY())
        K = jnp.zeros((self.num_controlled_vars))
        seed_points = jnp.vstack((x_coord, y_coord)).T
        
        # Create the grid points
        grid_points = jnp.vstack([X.ravel(), Y.ravel()]).T
        
        # Calculate Euclidean distance between each grid point and each seed point
        @jit
        def euclidean_distance(grid_point, seed_points):
            return jnp.sqrt(jnp.sum((grid_point - seed_points) ** 2, axis=1))
        
        # Iterate over grid points and assign the value from the nearest seed point
        @jit
        def assign_value_to_grid(grid_point):
            distances = euclidean_distance(grid_point, seed_points)
            nearest_seed_idx = jnp.argmin(distances)
            return k_values[nearest_seed_idx]

        assign_value_to_grid_vmap = vmap(assign_value_to_grid)
        K = assign_value_to_grid_vmap(grid_points)

        return K