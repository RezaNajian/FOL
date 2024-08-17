import sys
import os

import numpy as np
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.mechanical_3D_fe_tetra import MechanicalLoss3DTetra
from fol.solvers.fe_solver import FiniteElementSolver
from fol.IO.mesh_io import MeshIO
from fol.controls.displacement_control import DisplacementControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle


# directory & save handling
working_directory_name = "box_3D_tetra"
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# import mesh
point_bc_settings = {"Ux":{"left":0.0},
                    "Uy":{"left":0.0,"right":-0.05},
                    "Uz":{"left":0.0,"right":-0.05}}

io = MeshIO("fol_io",'../meshes/',"box_3D_coarse.med",point_bc_settings)
model_info = io.Import()

# creation of fe model and loss function
fe_model = FiniteElementModel("FE_model",model_info)
mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_loss_3d",fe_model,{"young_modulus":1,"poisson_ratio":0.3})

# fourier control
fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([2,4,6]),
                            "beta":20,"min":1e-1,"max":1}
fourier_control = DisplacementControl("fourier_control",fourier_control_settings,fe_model)


llll
# create some random coefficients & K for training
with open(f'fourier_control_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
coeffs_matrix = loaded_dict["coeffs_matrix"]
K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

eval_id = 69
io.mesh_io.point_data['K'] = np.array(K_matrix[eval_id,:])

# now we need to create, initialize and train fol
fol = FiniteElementOperatorLearning("first_fol",fourier_control,[mechanical_loss_3d],[1],
                                    "tanh",load_NN_params=False,working_directory=working_directory_name)
fol.Initialize()

fol_num_epochs = 2000
fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix[eval_id].reshape(-1,1).T,batch_size=1,num_epochs=fol_num_epochs,
            learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
            relative_error=1e-10,NN_params_save_file_name="NN_params_"+working_directory_name)

solution_file = os.path.join(case_dir, f"K_{eval_id}_comp.vtu")
FOL_UVW = np.array(fol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T))
io.mesh_io.point_data['U_FOL'] = FOL_UVW.reshape((fe_model.GetNumberOfNodes(), 3))

# solve FE here
solve_FE = False
if solve_FE:
    first_fe_solver = FiniteElementSolver("first_fe_solver",mechanical_loss_3d)
    FE_UVW = np.array(first_fe_solver.SingleSolve(K_matrix[eval_id],np.zeros(3*fe_model.GetNumberOfNodes())))  
    io.mesh_io.point_data['U_FE'] = FE_UVW.reshape((fe_model.GetNumberOfNodes(), 3))

io.mesh_io.write(solution_file)
