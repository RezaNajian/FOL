"""
 Authors: Yusuke Yamazaki
 Date: September, 2024
 License: FOL/License.txt
"""
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from  .fe_loss import FiniteElementLoss
from fol.tools.fem_utilities import *
from fol.computational_models.fe_model import FiniteElementModel

class ThermalLoss2D(FiniteElementLoss):
    """FE-based 2D Thermal loss

    This is the base class for the loss functions require FE formulation.

    """
    def __init__(self, name: str, fe_model: FiniteElementModel, loss_settings: dict={}):
        super().__init__(name,fe_model,["T"],{**loss_settings,"compute_dims":2,"rho":1.0,"cp":10.0, "dt":0.05})
        self.shape_function = QuadShapeFunction()
        self.rho = loss_settings["rho"]
        self.cp =  loss_settings["rho"]
        self.dt =  loss_settings["dt"]
        # self.Ke = loss_settings["Ke"]


    @partial(jit, static_argnums=(0,))
    def ComputeElement(self,xyze,Te_c,Te_n,body_force):
        xye = jnp.array([xyze[::3], xyze[1::3]])
        Te_c = Te_c.reshape(-1,1)
        Te_n = Te_n.reshape(-1,1)
        @jit
        def compute_at_gauss_point(xi,eta,total_weight):
            Nf = self.shape_function.evaluate(xi,eta)
            # conductivity_at_gauss = jnp.dot(Nf, Ke.squeeze())
            dN_dxi = self.shape_function.derivatives(xi,eta)
            J = jnp.dot(dN_dxi.T, xye.T)
            detJ = jnp.linalg.det(J)
            invJ = jnp.linalg.inv(J)
            B = jnp.dot(invJ,dN_dxi.T)
            T_at_gauss_n = jnp.dot(Nf, Te_n)
            T_at_gauss_c = jnp.dot(Nf, Te_c)
            gp_stiffness =  jnp.dot(B.T, B) * detJ * total_weight #* conductivity_at_gauss
            gp_mass =jnp.outer(Nf, Nf) * detJ * total_weight
            gp_f = total_weight * detJ * body_force *  Nf.reshape(-1,1) 
            gp_t = total_weight * detJ *(T_at_gauss_n-T_at_gauss_c)**2
            return gp_stiffness,gp_mass, gp_f, gp_t
        @jit
        def vmap_compatible_compute_at_gauss_point(gp_index):
            return compute_at_gauss_point(self.g_points[self.dim*gp_index],
                                          self.g_points[self.dim*gp_index+1],
                                          self.g_weights[self.dim*gp_index] * self.g_weights[self.dim*gp_index+1])

        k_gps,m_gps,f_gps,t_gps = jax.vmap(vmap_compatible_compute_at_gauss_point,(0))(jnp.arange(self.num_gp**self.dim))
        Se = jnp.sum(k_gps, axis=0)
        Me = jnp.sum(m_gps, axis=0)
        Fe = jnp.sum(f_gps, axis=0)
        Te = jnp.sum(t_gps)

        return  0.5*Te_n.T @Se@Te_n+0.5*self.rho*self.cp*Te/self.dt, (Me+self.dt*Se)@Te_n - Me@Te_c, (Me+self.dt*Se)
    
    def ComputeElementEnergy(self,xyze,de,uvwe,body_force=0.0):
        return self.ComputeElement(xyze,de,uvwe,body_force)[0]

    def ComputeElementResidualsAndStiffness(self,xyze,de,uvwe,body_force=0.0):
        _,re,ke = self.ComputeElement(xyze,de,uvwe,body_force)
        return re,ke

    def ComputeElementResiduals(self,xyze,de,uvwe,body_force=0.0):
        return self.ComputeElement(xyze,de,uvwe,body_force)[1]
    
    def ComputeElementStiffness(self,xyze,de,uvwe,body_force=0.0):
        return self.ComputeElement(xyze,de,uvwe,body_force)[2]

    @partial(jit, static_argnums=(0,))
    def ComputeElementResidualsVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,UV):
        return self.ComputeElementResiduals(jnp.ravel(jnp.column_stack((X[elements_nodes[element_id]],
                                                                     Y[elements_nodes[element_id]],
                                                                     Z[elements_nodes[element_id]]))),
                                                                     C[elements_nodes[element_id]],
                                                                     UV[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                     jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeElementResidualsAndStiffnessVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,UV):
        return self.ComputeElementResidualsAndStiffness(jnp.ravel(jnp.column_stack((X[elements_nodes[element_id]],
                                                                     Y[elements_nodes[element_id]],
                                                                     Z[elements_nodes[element_id]]))),
                                                                     C[elements_nodes[element_id]],
                                                                     UV[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                     jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    @partial(jit, static_argnums=(0,))
    def ComputeElementEnergyVmapCompatible(self,element_id,elements_nodes,X,Y,Z,C,UV):
        return self.ComputeElementEnergy(jnp.ravel(jnp.column_stack((X[elements_nodes[element_id]],
                                                                     Y[elements_nodes[element_id]],
                                                                     Z[elements_nodes[element_id]]))),
                                                                     C[elements_nodes[element_id]],
                                                                     UV[((self.number_dofs_per_node*elements_nodes[element_id])[:, jnp.newaxis] +
                                                                     jnp.arange(self.number_dofs_per_node))].reshape(-1,1))

    # @partial(jit, static_argnums=(0,))
    # def ComputeSingleLoss(self,full_control_params,unknown_dofs):
    #     elem_residual = self.ComputeResiduals(full_control_params.reshape(-1,1),
    #                                                   self.ExtendUnknowDOFsWithBC(unknown_dofs))
    #     # some extra calculation for reporting and not traced
    #     avg_elem_residual = jax.lax.stop_gradient(jnp.mean(elem_residual))
    #     max_elem_residual = jax.lax.stop_gradient(jnp.max(elem_residual))
    #     min_elem_residual = jax.lax.stop_gradient(jnp.min(elem_residual))
    #     return jnp.sum(elem_residual),(0,max_elem_residual,avg_elem_residual)

    @partial(jit, static_argnums=(0,))
    def ComputeSingleLoss(self,full_control_params,unknown_dofs):
        elems_energies = self.ComputeElementsEnergies(full_control_params.reshape(-1,1),
                                                      self.ExtendUnknowDOFsWithBC(unknown_dofs))
        # some extra calculation for reporting and not traced
        avg_elem_energy = jax.lax.stop_gradient(jnp.mean(elems_energies))
        max_elem_energy = jax.lax.stop_gradient(jnp.max(elems_energies))
        min_elem_energy = jax.lax.stop_gradient(jnp.min(elems_energies))
        return jnp.sum(elems_energies),(0,max_elem_energy,avg_elem_energy)
