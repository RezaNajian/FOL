"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2024
 License: FOL/License.txt
"""
from  .fe_loss import FiniteElementLoss
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.tools.fem_utilities import *
from fol.tools.decoration_functions import *
from fol.computational_models.fe_model import FiniteElementModel

class MechanicalLoss3DTetra(FiniteElementLoss):
    """FE-based Mechanical loss

    This is the base class for the loss functions require FE formulation.

    """
    @print_with_timestamp_and_execution_time
    def __init__(self, name: str, fe_model: FiniteElementModel, loss_settings: dict={}, dirichlet_bc_dict: dict={}):
        super().__init__(name,fe_model,["Ux","Uy","Uz"],{**loss_settings,"compute_dims":3}, dirichlet_bc_dict)
        self.shape_function = TetrahedralShapeFunction()

        # construction of the constitutive matrix
        self.e = self.loss_settings["young_modulus"]
        self.v = self.loss_settings["poisson_ratio"]
        c1 = self.e / ((1.0 + self.v) * (1.0 - 2.0 * self.v))
        c2 = c1 * (1.0 - self.v)
        c3 = c1 * self.v
        c4 = c1 * 0.5 * (1.0 - 2.0 * self.v)
        D = jnp.zeros((6,6))
        D = D.at[0,0].set(c2)
        D = D.at[0,1].set(c3)
        D = D.at[0,2].set(c3)
        D = D.at[1,0].set(c3)
        D = D.at[1,1].set(c2)
        D = D.at[1,2].set(c3)
        D = D.at[2,0].set(c3)
        D = D.at[2,1].set(c3)
        D = D.at[2,2].set(c2)
        D = D.at[3,3].set(c4)
        D = D.at[4,4].set(c4)
        D = D.at[5,5].set(c4)
        self.D = D

    @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,de,uvwe,body_force):
        xyze = jnp.array([xyze[::3], xyze[1::3], xyze[2::3]])
        @jit
        def compute_at_gauss_point(xi,eta,zeta,total_weight):
            Nf = self.shape_function.evaluate(xi,eta,zeta)
            e_at_gauss = jnp.dot(Nf, de.squeeze())
            dN_dxi = self.shape_function.derivatives(xi,eta,zeta)
            J = jnp.dot(dN_dxi.T, xyze.T)
            detJ = jnp.linalg.det(J)
            invJ = jnp.linalg.inv(J)
            dN_dX = jnp.dot(invJ,dN_dxi.T)
            B = jnp.zeros((6,uvwe.size))
            index = jnp.arange(4) * 3
            B = B.at[0, index + 0].set(dN_dX[0, :])
            B = B.at[1, index + 1].set(dN_dX[1, :])
            B = B.at[2, index + 2].set(dN_dX[2, :])
            B = B.at[3, index + 0].set(dN_dX[1, :])
            B = B.at[3, index + 1].set(dN_dX[0, :])
            B = B.at[4, index + 1].set(dN_dX[2, :])
            B = B.at[4, index + 2].set(dN_dX[1, :])
            B = B.at[5, index + 0].set(dN_dX[2, :])
            B = B.at[5, index + 2].set(dN_dX[0, :])
            N = jnp.zeros((3,uvwe.size))
            N = N.at[0,0::3].set(Nf)
            N = N.at[0,1::3].set(Nf)
            N = N.at[0,2::3].set(Nf)
            gp_stiffness = total_weight * detJ * e_at_gauss * (B.T @ (self.D @ B))
            gp_f = total_weight * detJ * (N.T @ body_force)
            return gp_stiffness,gp_f
        @jit
        def vmap_compatible_compute_at_gauss_point(gp_index):
            return compute_at_gauss_point(self.g_points[self.dim*gp_index],
                                          self.g_points[self.dim*gp_index+1],
                                          self.g_points[self.dim*gp_index+2],
                                          self.g_weights[self.dim*gp_index] * 
                                          self.g_weights[self.dim*gp_index+1]* 
                                          self.g_weights[self.dim*gp_index+2])

        k_gps,f_gps = jax.vmap(vmap_compatible_compute_at_gauss_point,(0))(jnp.arange(self.num_gp**self.dim))
        Se = jnp.sum(k_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        element_residuals = jax.lax.stop_gradient(Se @ uvwe - Fe)
        return  ((uvwe.T @ element_residuals)[0,0]), 2 * (Se @ uvwe - Fe), 2 * Se

    def ComputeElementEnergy(self,xyze,de,uvwe,body_force=jnp.zeros((3,1))):
        return self.ComputeElement(xyze,de,uvwe,body_force)[0]

    def ComputeElementResidualsAndStiffness(self,xyze,de,uvwe,body_force=jnp.zeros((3,1))):
        _,re,ke = self.ComputeElement(xyze,de,uvwe,body_force)
        return re,ke

    def ComputeElementResiduals(self,xyze,de,uvwe,body_force=jnp.zeros((3,1))):
        return self.ComputeElement(xyze,de,uvwe,body_force)[1]
    
    def ComputeElementStiffness(self,xyze,de,uvwe,body_force=jnp.zeros((3,1))):
        return self.ComputeElement(xyze,de,uvwe,body_force)[2]

    @partial(jit, static_argnums=(0,))
    def ComputeElementResidualsVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,UVW):
        return self.ComputeElementResiduals(jnp.ravel(jnp.column_stack((X[elements_nodes[element_id]],
                                                                     Y[elements_nodes[element_id]],
                                                                     Z[elements_nodes[element_id]]))),
                                                                     C[elements_nodes[element_id]],
                                                                     UVW[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                     jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeElementResidualsAndStiffnessVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,UVW):
        return self.ComputeElementResidualsAndStiffness(jnp.ravel(jnp.column_stack((X[elements_nodes[element_id]],
                                                                     Y[elements_nodes[element_id]],
                                                                     Z[elements_nodes[element_id]]))),
                                                                     C[elements_nodes[element_id]],
                                                                     UVW[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                     jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeElementEnergyVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,UVW):
        return self.ComputeElementEnergy(jnp.ravel(jnp.column_stack((X[elements_nodes[element_id]],
                                                                     Y[elements_nodes[element_id]],
                                                                     Z[elements_nodes[element_id]]))),
                                                                     C[elements_nodes[element_id]],
                                                                     UVW[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                     jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeSingleLoss(self,known_dofs,unknown_dofs):
        full_UVW = self.GetFullDofVector(known_dofs,unknown_dofs)
        psudo_k = jnp.ones(full_UVW.shape)
        elems_energies = self.ComputeElementsEnergies(psudo_k.reshape(-1,1),
                                                      full_UVW)
        # some extra calculation for reporting and not traced
        avg_elem_energy = jax.lax.stop_gradient(jnp.mean(elems_energies))
        max_elem_energy = jax.lax.stop_gradient(jnp.max(elems_energies))
        min_elem_energy = jax.lax.stop_gradient(jnp.min(elems_energies))
        return jnp.abs(jnp.sum(elems_energies)),(0,max_elem_energy,avg_elem_energy)

