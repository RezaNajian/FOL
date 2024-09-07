"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from  .loss import Loss
import jax
import jax.numpy as jnp
import warnings
from jax import jit,grad
from functools import partial
from abc import abstractmethod
from fol.tools.decoration_functions import *
from fol.computational_models.fe_model import FiniteElementModel
from fol.tools.fem_utilities import *

class FiniteElementLoss(Loss):
    """FE-based losse

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, fe_model: FiniteElementModel, ordered_dofs: list, loss_settings: dict, dirichlet_bc_dict: dict):
        super().__init__(name)
        self.fe_model = fe_model
        self.dofs = ordered_dofs
        self.loss_settings = loss_settings
        self.number_dofs_per_node = len(self.dofs)
        self.dirichlet_bc_dict = dirichlet_bc_dict
        self.total_number_of_dofs = len(self.dofs) * self.fe_model.GetNumberOfNodes()
        self.dof_dict = self.fe_model.GetDofsDict(self.dofs,self.dirichlet_bc_dict)
        self.non_dirichlet_indices = self.dof_dict["non_dirichlet_indices"]
        self.dirichlet_indices = self.dof_dict["dirichlet_indices"]
        self.dirichlet_values = self.dof_dict["dirichlet_values"]
        self.number_of_unknown_dofs = self.non_dirichlet_indices.size

        # create full solution vector
        self.solution_vector = jnp.zeros(self.total_number_of_dofs)
        # apply dirichlet bcs
        self.solution_vector = self.solution_vector.at[self.dirichlet_indices].set(self.dirichlet_values)

        # now prepare gauss integration
        if "num_gp" in self.loss_settings.keys():
            self.num_gp = self.loss_settings["num_gp"]
            if self.num_gp == 1:
                g_points,g_weights = GaussQuadrature().one_point_GQ
            elif self.num_gp == 2:
                g_points,g_weights = GaussQuadrature().two_point_GQ
            elif self.num_gp == 3:
                g_points,g_weights = GaussQuadrature().three_point_GQ
            elif self.num_gp == 4:
                g_points,g_weights = GaussQuadrature().four_point_GQ
            else:
                raise ValueError(f" number gauss points {self.num_gp} is not supported ! ")
        else:
            g_points,g_weights = GaussQuadrature().one_point_GQ
            self.loss_settings["num_gp"] = 1
            self.num_gp = 1
            warnings.warn(f"number of gauss points is set to 1 for loss {self.GetName()}!")

        if not "compute_dims" in self.loss_settings.keys():
            raise ValueError(f"compute_dims must be provided in the loss settings of {self.GetName()}! ")

        self.dim = self.loss_settings["compute_dims"]

        if self.dim==1:
            self.g_points = jnp.array([[xi] for xi in g_points]).flatten()
            self.g_weights = jnp.array([[w_i] for w_i in g_weights]).flatten()
        elif self.dim==2:
            self.g_points = jnp.array([[xi, eta] for xi in g_points for eta in g_points]).flatten()
            self.g_weights = jnp.array([[w_i , w_j] for w_i in g_weights for w_j in g_weights]).flatten()
        elif self.dim==3:
            self.g_points = jnp.array([[xi,eta,zeta] for xi in g_points for eta in g_points for zeta in g_points]).flatten()
            self.g_weights = jnp.array([[w_i,w_j,w_k] for w_i in g_weights for w_j in g_weights for w_k in g_weights]).flatten()

    def GetLossDofsDict(self) -> dict:
        return self.dof_dict

    def Initialize(self) -> None:
        pass

    def Finalize(self) -> None:
        pass

    def GetNumberOfUnknowns(self):
        return self.number_of_unknown_dofs

    @abstractmethod
    def ComputeElementEnergy(self):
        pass

    @abstractmethod
    def ComputeElementResiduals(self):
        pass

    @abstractmethod
    def ComputeElementStiffness(self):
        pass

    @abstractmethod
    def ComputeElementResidualsAndStiffness(self):
        pass

    @abstractmethod
    def ComputeElementEnergyVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,P):
        pass

    @abstractmethod
    def ComputeElementResidualsVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,P):
        pass

    @partial(jit, static_argnums=(0,))
    def ComputeElementsEnergies(self,total_control_vars,total_primal_vars):
        # parallel calculation of energies
        return jax.vmap(self.ComputeElementEnergyVmapCompatible,(0,None,None,None,None,None,None)) \
                        (self.fe_model.GetElementsIds(),self.fe_model.GetElementsNodes()
                        ,self.fe_model.GetNodesX(),self.fe_model.GetNodesY(),self.fe_model.GetNodesZ(),
                        total_control_vars,total_primal_vars)

    @partial(jit, static_argnums=(0,))
    def ComputeTotalEnergy(self,total_control_vars,total_primal_vars):
        return jnp.sum(self.ComputeElementsEnergies(total_control_vars,total_primal_vars))

    @print_with_timestamp_and_execution_time
    def ComputeResiduals(self,total_control_vars,total_primal_vars):
        return jax.grad(self.ComputeTotalEnergy,argnums=1)(total_control_vars,total_primal_vars)
    
    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,))
    def ComputeResidualsAndStiffness(self,total_control_vars,total_primal_vars):
        psudo_k = jnp.ones(int(total_primal_vars.shape[0]/3))
        residuals = self.ComputeResiduals(psudo_k,total_primal_vars)
        stiffness = jnp.squeeze(jax.jacfwd(self.ComputeResiduals,argnums=1)(psudo_k,total_primal_vars))
        return residuals,stiffness
    
    @partial(jit, static_argnums=(0,))
    def Compute_DR_DC(self,total_control_vars,total_primal_vars):
        return jax.jacfwd(self.Compute_R,argnums=0)(total_control_vars,total_primal_vars)
    
    @partial(jit, static_argnums=(0,))
    def ExtendUnknowDOFsWithBC(self,unknown_dofs):
        self.solution_vector = self.solution_vector.at[self.non_dirichlet_indices].set(unknown_dofs)
        return self.solution_vector
    
    @partial(jit, static_argnums=(0,))
    def GetFullDofVector(self,known_dofs,unknown_dofs):
        self.solution_vector = self.solution_vector.at[self.dirichlet_indices].set(known_dofs)
        self.solution_vector = self.solution_vector.at[self.non_dirichlet_indices].set(unknown_dofs)
        return self.solution_vector
    
    @partial(jit, static_argnums=(0,))
    def ApplyBCOnR(self,full_residual_vector):
        return full_residual_vector.at[self.dirichlet_indices].set(0.0)
    
    @partial(jit, static_argnums=(0,))
    def ApplyBCOnMatrix(self,full_matrix):
        full_matrix = full_matrix.at[self.dirichlet_indices,:].set(0)
        full_matrix = full_matrix.at[self.dirichlet_indices,self.dirichlet_indices].set(1)
        return full_matrix

    @partial(jit, static_argnums=(0))
    def ApplyBCOnDOFs(self,known_dofs,full_dof_vector,load_increment=1):
        full_dof_vector = full_dof_vector.reshape(-1)
        return full_dof_vector.at[self.dirichlet_indices].set(known_dofs)
            


