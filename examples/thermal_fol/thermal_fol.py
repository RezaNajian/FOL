import os,time,sys
import numpy as np
from fol.IO.mesh_io import MeshIO
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.thermal_3D_fe_tetra_sens import ThermalLoss3DTetraSens
from fol.loss_functions.thermal_3D_fe_tetra_res import ThermalLoss3DTetraRes
from fol.loss_functions.thermal_3D_fe_tetra_en import ThermalLoss3DTetraEnergy
from fol.solvers.nonlinear_solver import NonLinearSolver
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle

# cleaning & logging
working_directory_name = 'thermal_fol'
case_dir = os.path.join('.', working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,"fol_thermal_3D.log"))

# importing mesh & creating model info
point_bc_settings = {"T":{"left_fol":0.1,"right_fol":0.01}}
io = MeshIO("fol_io",'../meshes/',"fol_3D_tet_mesh_coarse.med",point_bc_settings)
model_info = io.Import()

# create FE model
fe_model = FiniteElementModel("FE_model",model_info)

# create thermal loss
thermal_loss_3d_res = ThermalLoss3DTetraRes("thermal_loss_3d_residual",fe_model,{"beta":2,"c":4})
thermal_loss_3d_en = ThermalLoss3DTetraEnergy("thermal_loss_3d_energy",fe_model,{"beta":2,"c":4})
thermal_loss_3d_sens = ThermalLoss3DTetraSens("thermal_loss_3d_residual_sens",fe_model,{"beta":2,"c":4})

# create Fourier parametrization/control
x_freqs = np.array([1,2,3])
y_freqs = np.array([1,2,3])
z_freqs = np.array([0])
fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":5,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_model)

with open(f'fourier_control_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

coeffs_matrix = loaded_dict["coeffs_matrix"]
K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

# specify id of the K of interest
eval_id = 5
io.mesh_io.point_data['K'] = np.array(K_matrix[eval_id,:])

# now we need to create, initialize and train fol
fol = FiniteElementOperatorLearning("first_fol",fourier_control,[thermal_loss_3d_en,thermal_loss_3d_sens],[],
                                    "swish",load_NN_params=False,working_directory=working_directory_name)

fol.Initialize()
fol_num_epochs = 2000
fol.Train(loss_functions_weights=[1,1],X_train=coeffs_matrix[eval_id].reshape(-1,1).T,batch_size=1,num_epochs=fol_num_epochs,
          learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",relative_error=1e-200,absolute_error=1e-200,
          NN_params_save_file_name="NN_params_"+working_directory_name)

solution_file = os.path.join(case_dir, f"K_{eval_id}_comp.vtu")
FOL_T = np.array(fol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T))
io.mesh_io.point_data['T_FOL'] = FOL_T.reshape((fe_model.GetNumberOfNodes(), 1))

io.mesh_io.write(solution_file)

# solve FE here
solve_FE = True
if solve_FE: 
    first_fe_solver = NonLinearSolver("first_fe_solver",thermal_loss_3d_en,relative_error=1e-5,max_num_itr=20,load_incr=1)
    start_time = time.process_time()
    FE_T = np.array(first_fe_solver.SingleSolve(K_matrix[eval_id],np.zeros(fe_model.GetNumberOfNodes())))  
    print(f"\n############### FE solve took: {time.process_time() - start_time} s ###############\n")
    io.mesh_io.point_data['T_FE'] = FE_T.reshape((fe_model.GetNumberOfNodes(), 1))

    relative_error = 100 * (abs(FOL_T.reshape(-1,1)-FE_T.reshape(-1,1)))/abs(FE_T.reshape(-1,1))
    io.mesh_io.point_data['relative_error'] = relative_error.reshape((fe_model.GetNumberOfNodes(), 1))

io.mesh_io.write(solution_file)
