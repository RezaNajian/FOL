"""
 Authors: Kianoosh Taghikhani, https://github.com/kianoosh1989
 Date: July, 2024
 License: FOL/License.txt
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit,jacfwd
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

        # if isinstance(self.settings["k_rangeof_values"],tuple):
        #     start, end = self.settings["k_rangeof_values"]
        #     self.k_rangeof_values = self.settings["k_rangeof_values"]
        # if isinstance(self.settings["k_rangeof_values"],list):
        #     self.k_rangeof_values = self.settings["k_rangeof_values"]

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
        # Split the input vector into x, y coordinates and k_values
        x_coord = variable_vector[:self.numberof_seeds]
        y_coord = variable_vector[self.numberof_seeds:2*self.numberof_seeds]
        k_values = variable_vector[2*self.numberof_seeds:]
        # Define grid based on the number of controlled variables
        # N = int(self.num_controlled_vars**0.5)
        # x = np.linspace(0, 1, N)
        # y = np.linspace(0, 1, N)
        # X, Y = np.meshgrid(x, y)
        X = np.array(self.fe_model.GetNodesX())
        Y = np.array(self.fe_model.GetNodesY())
        # Initialize array for controlled variables
        K = jnp.zeros((self.num_controlled_vars))
        # Combine seed points and ensure they have correct size
        seed_points = jnp.vstack((x_coord, y_coord)).T.astype(float)
        if seed_points.shape[0] < 4:
            raise ValueError("At least 4 seed points are required to create a Voronoi diagram.")
        if x_coord.shape[-1] != self.numberof_seeds or y_coord.shape[-1] != self.numberof_seeds or k_values.shape[-1] != self.numberof_seeds:
            raise ValueError("Number of coordinates should be equal to number of seed points!")

        # Add a small perturbation to avoid coplanar issues
        random_seed: int = 42
        key = PRNGKey(random_seed)
        perturbation = normal(key, shape=seed_points.shape) * 1e-8
        seed_points += perturbation
        # Convert JAX array to NumPy array for KDTree
        seed_points_np = np.random.normal(scale=1e-8, size=seed_points.shape)
        # Use KDTree to assign nearest seed points
        tree = KDTree(seed_points_np)
        grid_points = np.vstack([X.flatten(), Y.flatten()]).T
        # Find nearest seed point for each grid point
        _, regions = tree.query(grid_points)
        # Assign the feature value based on the nearest seed point
        for i, region in enumerate(regions):
            K = K.at[i].set(k_values[region])
        return K


    @partial(jit, static_argnums=(0,))
    def ComputeJacobian(self,control_vec):
        return jnp.squeeze(jacfwd(self.ComputeControlledVariables,argnums=0)(control_vec))