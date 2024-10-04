"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit,jacfwd,vmap
from functools import partial
from jax.nn import sigmoid
from fol.mesh_input_output.mesh import Mesh
from fol.tools.decoration_functions import *

class FourierControl(Control):
    
    def __init__(self,control_name: str,control_settings: dict, fe_mesh: Mesh):
        super().__init__(control_name)
        self.settings = control_settings
        self.fe_mesh = fe_mesh

    def GetNumberOfVariables(self):
        return self.num_control_vars
    
    def GetNumberOfControlledVariables(self):
        return self.num_controlled_vars

    @print_with_timestamp_and_execution_time
    def Initialize(self) -> None:
        if "min" in self.settings.keys():
            self.min = self.settings["min"]
        else:
            self.min = 1e-6
        if "max" in self.settings.keys():
            self.max = self.settings["max"]
        else:
            self.max = 1.0
        self.beta = self.settings["beta"]
        self.x_freqs = self.settings["x_freqs"]
        self.y_freqs = self.settings["y_freqs"]
        self.z_freqs = self.settings["z_freqs"]
        self.num_x_freqs = self.x_freqs.shape[-1]
        self.num_y_freqs = self.y_freqs.shape[-1]
        self.num_z_freqs = self.z_freqs.shape[-1]
        self.num_control_vars = self.num_x_freqs * self.num_y_freqs * self.num_z_freqs + 1
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()
        mesh_x, mesh_y, mesh_z = jnp.meshgrid(self.x_freqs, self.y_freqs, self.z_freqs, indexing='ij')
        self.frquencies_vec = jnp.vstack([mesh_x.ravel(), mesh_y.ravel(), mesh_z.ravel()]).T
        self.initialized = True

    def Finalize(self) -> None:
        pass

    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self,variable_vector:jnp.array):
        @jit
        def evaluate_at_frequencies(freqs,coeff):
            cos_x = jnp.cos(freqs[0] * jnp.pi * self.fe_mesh.GetNodesX())
            cos_y = jnp.cos(freqs[1] * jnp.pi * self.fe_mesh.GetNodesY())
            cos_z = jnp.cos(freqs[2] * jnp.pi * self.fe_mesh.GetNodesZ())
            return coeff * cos_x * cos_y * cos_z
        K = (variable_vector[0]/2.0) + jnp.sum(vmap(evaluate_at_frequencies, in_axes=(0, 0))
                                               (self.frquencies_vec,variable_vector[1:]),axis=0)
        return (self.max-self.min) * sigmoid(self.beta*(K-0.5)) + self.min
    
    @partial(jit, static_argnums=(0,))
    def ComputeJacobian(self,control_vec):
        return jnp.squeeze(jacfwd(self.ComputeControlledVariables,argnums=0)(control_vec))