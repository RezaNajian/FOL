
import numpy as np
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.mechanical_3D_fe_tetra import MechanicalLoss3DTetra
from fol.IO.mdpa_io import MdpaIO
from fol.controls.dirichlet_condition_control import DirichletConditionControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle
from fol.solvers.fe_solver import FiniteElementSolver

# directory & save handling
working_directory_name = "results"
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

point_bc_settings = {"Ux":{"DISPLACEMENT_Displacement_Auto3":0.0,"DISPLACEMENT_Displacement_Auto4":0.01},
                     "Uy":{"DISPLACEMENT_Displacement_Auto3":0.0},
                     "Uz":{"DISPLACEMENT_Displacement_Auto3":0.0}}

mdpa_io = MdpaIO("mdpa_io","patch_test_3D_tension_tetra.mdpa",bc_settings=point_bc_settings,scale_factor=1.0)
model_info = mdpa_io.Import()

# creation of fe model and loss function
fe_model = FiniteElementModel("FE_model",model_info,mdpa_io)

# create loss
mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_loss_3d",fe_model,{"young_modulus":1,"poisson_ratio":0.3},point_bc_settings)

# displ_control_settings = {"Ux":["DISPLACEMENT_Displacement_Auto4"]}
# displ_control = DirichletConditionControl("displ_control",displ_control_settings,mechanical_loss_3d)

# # create some random bcs 
# create_random_coefficients = False
# if create_random_coefficients:
#     number_of_random_samples = 2
#     bc_matrix,bc_nodal_value_matrix = create_normal_dist_bc_samples(displ_control,
#                                                                     numberof_sample=number_of_random_samples,
#                                                                     center=-0.5,standard_dev=0.25)
#     export_dict = {}
#     export_dict["bc_matrix"] = bc_matrix
#     export_dict["point_bc_settings"] = point_bc_settings
#     export_dict["displ_control_settings"] = displ_control_settings
#     with open(f'bc_control_dict.pkl', 'wb') as f:
#         pickle.dump(export_dict,f)
# else:
#     with open(f'bc_control_dict.pkl', 'rb') as f:
#         loaded_control_dict = pickle.load(f)
    
#     bc_matrix = loaded_control_dict["bc_matrix"]

# bc_nodal_value_matrix = displ_control.ComputeBatchControlledVariables(bc_matrix)

# # export generated bcs
# export_bcs = False
# if export_bcs:
#     num_unknowns = mechanical_loss_3d.GetNumberOfUnknowns()
#     unknown_dofs = jnp.zeros(num_unknowns)
#     for i in range(bc_nodal_value_matrix.shape[0]):
#         full_displ_vec = mechanical_loss_3d.GetFullDofVector(bc_nodal_value_matrix[i,:],unknown_dofs)
#         mdpa_io[f'bc_{i}'] = np.array(full_displ_vec).reshape((fe_model.GetNumberOfNodes(), 3))
        
#     mdpa_io.Export(export_dir=case_dir)

eval_id = 0
# print(bc_nodal_value_matrix)
first_fe_solver = FiniteElementSolver("first_fe_solver",mechanical_loss_3d)
FE_UVW = np.array(first_fe_solver.SingleSolve(jnp.zeros(3*fe_model.GetNumberOfNodes())))  
mdpa_io[f'U_FE'] = FE_UVW.reshape((fe_model.GetNumberOfNodes(), 3))

mdpa_io.Export(export_dir=case_dir)


exit()
# now we need to create, initialize and train fol
fol = FiniteElementOperatorLearning("first_fol",displ_control,[mechanical_loss_3d],[10,10],
                                    "swish",load_NN_params=False,working_directory=working_directory_name)
# now create train and test samples
num_train_samples = int(0.8 * bc_matrix.shape[0])
bc_train_mat = bc_matrix[0:num_train_samples]
bc_train_nodal_value_matrix = bc_nodal_value_matrix[0:num_train_samples]

bc_test_mat = bc_matrix[num_train_samples:]
bc_test_nodal_value_matrix = bc_nodal_value_matrix[num_train_samples:]

fol.Initialize()
fol_num_epochs = 2000
fol_batch_size = 1
fol_learning_rate = 0.0005
fol.Train(loss_functions_weights=[1],X_train=bc_train_mat,batch_size=fol_batch_size,
          num_epochs=fol_num_epochs,learning_rate=fol_learning_rate,optimizer="adam",
          convergence_criterion="total_loss",relative_error=1e-10,
          NN_params_save_file_name="NN_params_"+working_directory_name)

UVW_train = fol.Predict(bc_train_mat)
UVW_test = fol.Predict(bc_test_mat)

test_eval_ids = [0,10,15]
for eval_id in test_eval_ids:
    num_unknowns = mechanical_loss_3d.GetNumberOfUnknowns()
    unknown_dofs = jnp.zeros(num_unknowns)
    full_displ_bc_vec = mechanical_loss_3d.GetFullDofVector(bc_test_nodal_value_matrix[eval_id,:],unknown_dofs)
    mdpa_io[f'test_bc_{eval_id}'] = np.array(full_displ_bc_vec).reshape((fe_model.GetNumberOfNodes(), 3))
    mdpa_io[f'test_U_FOL_{eval_id}'] = np.array(UVW_test[eval_id]).reshape((fe_model.GetNumberOfNodes(), 3))

train_eval_ids = [0,1,2]
for eval_id in train_eval_ids:
    num_unknowns = mechanical_loss_3d.GetNumberOfUnknowns()
    unknown_dofs = jnp.zeros(num_unknowns)
    full_displ_bc_vec = mechanical_loss_3d.GetFullDofVector(bc_train_nodal_value_matrix[eval_id,:],unknown_dofs)
    mdpa_io[f'train_bc_{eval_id}'] = np.array(full_displ_bc_vec).reshape((fe_model.GetNumberOfNodes(), 3))
    mdpa_io[f'train_U_FOL_{eval_id}'] = np.array(UVW_train[eval_id]).reshape((fe_model.GetNumberOfNodes(), 3))

mdpa_io.Export(export_dir=case_dir)
