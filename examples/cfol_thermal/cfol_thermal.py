import os,sys,pickle, time
import numpy as np
from fol.IO.mesh_io import MeshIO
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.thermal_2D_fe_quad_cont import ThermalLoss2DCont
from fol.solvers.fe_solver import FiniteElementSolver
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.implicit_operator_learning import ImplicitOperatorLearning
from fol.tools.decoration_functions import *
from fol.tools.logging_functions import Logger
from fol.tools.usefull_functions import *


# cleaning & logging
working_directory_name = 'cfol_results'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,"terminal.log"))

model_info = create_2D_square_model_info_thermal(L=1,N=51,T_left=1.0,T_right=0.1)
fe_model = FiniteElementModel("FE_model",model_info)
thermal_loss = ThermalLoss2DCont("thermal_loss",fe_model,{"num_gp":2})

fe_solver = FiniteElementSolver("fe_solver",thermal_loss)
fourier_control_settings = {"x_freqs":np.array([1,2,3]),"y_freqs":np.array([1,2,3]),"z_freqs":np.array([0]),"beta":2}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_model)

# create some random coefficients & K for training
create_random_coefficients = False
if create_random_coefficients:
    number_of_random_samples = 200
    coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
    export_dict = {}
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["x_freqs"] = fourier_control.x_freqs
    export_dict["y_freqs"] = fourier_control.y_freqs
    export_dict["z_freqs"] = fourier_control.z_freqs
    with open(f'fourier_control_dict.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(f'fourier_control_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]
    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

sample_index = 1
cfol = ImplicitOperatorLearning("first_cfol",loss_functions=[thermal_loss],
                                hidden_layers=[50,50],conductivity=K_matrix[sample_index],
                                omega=30,working_directory=case_dir)

cfol.Initialize()
cfol.Train(loss_functions_weights=[1],X_train=fe_model.GetNodesCoordinatesMatrix(),
           batch_size=fe_model.GetNumberOfNodes(),num_epochs=2000,
           learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",relative_error=1e-6)

T_cFOL = np.array(cfol.Predict(fe_model.GetNodesCoordinatesMatrix()))

T_FEM = np.array(fe_solver.SingleSolve(K_matrix[sample_index,:],np.zeros(fe_model.GetNumberOfNodes())))

l2_error = 100 * (abs(T_cFOL.reshape(-1,1)-T_FEM.reshape(-1,1)))/abs(T_FEM.reshape(-1,1))

plot_mesh_vec_data(1,[K_matrix[sample_index],T_cFOL,T_FEM,l2_error],["K","T_cFOL","T_FEM","relative error (%)"],file_name=os.path.join(case_dir,"cFOL-KT-dist.png"))
