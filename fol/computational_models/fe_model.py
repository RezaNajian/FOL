"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from  .model import Model
from fol.IO.input_output import InputOutput
from fol.tools.decoration_functions import *
import jax.numpy as jnp

class FiniteElementModel(Model):
    """Base abstract model class.

    The base abstract control class has the following responsibilities.
        1. Initalizes and finalizes the model.

    """
    @print_with_timestamp_and_execution_time
    def __init__(self, model_name: str, model_info: dict, model_io: InputOutput=None) -> None:
        super().__init__(model_name,model_io)
        self.nodes_dict = model_info["nodes_dict"]
        self.total_number_nodes = self.nodes_dict["nodes_ids"].shape[-1] 
        self.elements_dict = model_info["elements_dict"]
        self.total_number_elements = self.elements_dict["elements_ids"].shape[-1]
        self.dofs_dict = model_info["dofs_dict"]  

    def Initialize(self) -> None:
        pass

    def GetNumberOfNodes(self):
        return self.total_number_nodes

    def GetNumberOfElements(self):
        return self.total_number_elements

    def GetNodesIds(self):
        return self.nodes_dict["nodes_ids"]    

    def GetElementsIds(self):
        return self.elements_dict["elements_ids"]
    
    def GetElementsNodes(self):
        return self.elements_dict["elements_nodes"]
    
    def GetNodesCoordinates(self):
        return self.nodes_dict["X"],self.nodes_dict["Y"],self.nodes_dict["Z"]
    
    def GetNodesX(self):
        return self.nodes_dict["X"]
    
    def GetNodesY(self):
        return self.nodes_dict["Y"]
    
    def GetNodesZ(self):
        return self.nodes_dict["Z"]

    def GetDofsDict(self, dofs_list:list, dirichlet_bc_dict:dict):
        number_dofs_per_node = len(dofs_list)
        dirichlet_indices = []
        dirichlet_values = []        
        point_sets = self.model_io.GetPointSets()
        for dof_index,dof in enumerate(dofs_list):
            for boundary_name,boundary_value in dirichlet_bc_dict[dof].items():
                boundary_node_ids = jnp.array(point_sets[boundary_name])
                dirichlet_bc_indices = number_dofs_per_node*boundary_node_ids + dof_index
                dirichlet_indices.extend(dirichlet_bc_indices.tolist())
                dirichlet_bc_values = boundary_value * jnp.ones(dirichlet_bc_indices.size)
                dirichlet_values.extend(dirichlet_bc_values.tolist())
        
        dirichlet_indices = jnp.array(dirichlet_indices)
        dirichlet_values = jnp.array(dirichlet_values)
        all_indices = jnp.arange(number_dofs_per_node*self.GetNumberOfNodes())
        non_dirichlet_indices = jnp.setdiff1d(all_indices, dirichlet_indices)
        return  {"dirichlet_indices":dirichlet_indices,
                 "dirichlet_values":dirichlet_values,
                 "non_dirichlet_indices":non_dirichlet_indices}
    
    def Finalize(self) -> None:
        pass



