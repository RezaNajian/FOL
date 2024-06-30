import sys
import os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import numpy as np
from IO import MeshIO
from computational_models import FiniteElementModel
from loss_functions import ThermalLoss3DTetra
from solvers import FiniteElementSolver
from controls import FourierControl
from deep_neural_networks import FiniteElementOperatorLearning
from tools import *
import pickle

working_directory_name = 'fol_thermal_3D'
case_dir = os.path.join('.', working_directory_name)
clean_dir = True
if clean_dir:
    create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,"fol_thermal_3D.log"))

point_bc_settings = {"T":{"left_fol":1,"right_fol":0}}
io = MeshIO("fol_io",'.',"FOL_3D_tet.med",point_bc_settings)
model_info = io.Import()

# creation of the objects
fe_model = FiniteElementModel("FE_model",model_info)

thermal_loss_3d = ThermalLoss3DTetra("thermal_loss_3d",fe_model)

# fourier freqs
x_freqs = np.array([2,4,6])
y_freqs = np.array([2,4,6])
z_freqs = np.array([2,4,6])
fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":20,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_model)

# create some random coefficients & K for training
create_random_coefficients = True
if create_random_coefficients:
    number_of_random_samples = 200
    coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
    export_dict = {}
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["x_freqs"] = x_freqs
    export_dict["y_freqs"] = y_freqs
    export_dict["z_freqs"] = z_freqs
#     with open(f'fourier_control_dict.pkl', 'wb') as f:
#         pickle.dump(export_dict,f)
# else:
#     with open(f'fourier_control_dict.pkl', 'rb') as f:
#         loaded_dict = pickle.load(f)
    
    # coeffs_matrix = loaded_dict["coeffs_matrix"]

K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

eval_id = 69

# now we need to create, initialize and train fol
fol = FiniteElementOperatorLearning("first_fol",fourier_control,[thermal_loss_3d],[1],
                                    "tanh",load_NN_params=False,working_directory=working_directory_name)
fol.Initialize()

fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix[eval_id].reshape(-1,1).T,batch_size=1,num_epochs=2000,
            learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
            relative_error=1e-10,NN_params_save_file_name="NN_params_"+working_directory_name)

io.mesh_io.point_data['K'] = np.array(K_matrix[eval_id,:])

solution_file = os.path.join(case_dir, f"K_{eval_id}_comp.vtu")
FOL_T = np.array(fol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T))
io.mesh_io.point_data['T_FOL'] = FOL_T.reshape((fe_model.GetNumberOfNodes(), 1))

# first_fe_solver = FiniteElementSolver("first_fe_solver",thermal_loss_3d)
# FE_T = np.array(first_fe_solver.SingleSolve(K_matrix[eval_id],np.zeros(fe_model.GetNumberOfNodes())))  
# io.mesh_io.point_data['T_FE'] = FE_T.reshape((fe_model.GetNumberOfNodes(), 1))

io.mesh_io.write(solution_file)