
import numpy as np
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.mechanical_3D_fe_tetra import MechanicalLoss3DTetra
from fol.IO.mdpa_io import MdpaIO
from fol.controls.dirichlet_condition_control import DirichletConditionControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *
from fol.solvers.fe_solver import FiniteElementSolver
from fol.tools.logging_functions import *
import pickle

# directory & save handling
working_directory_name = "fol_results"
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

point_bc_settings = {"Ux":{"fixed":0.0},
                     "Uy":{"fixed":0.0},
                     "Uz":{"fixed":0.0,"moving":0.0}}
mdpa_io = MdpaIO("mdpa_io","solid.mdpa",bc_settings=point_bc_settings,scale_factor=1.0)
model_info = mdpa_io.Import()

# creation of fe model and loss function
fe_model = FiniteElementModel("FE_model",model_info,mdpa_io)

# create loss
mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_loss_3d",fe_model,{"young_modulus":1,"poisson_ratio":0.3,"num_gp":1},point_bc_settings)

displ_control_settings = {"Ux":[],
                          "Uy":[],
                          "Uz":["moving"]}
displ_control = DirichletConditionControl("displ_control",displ_control_settings,mechanical_loss_3d)

# create some random bcs 
create_random_coefficients = False
if create_random_coefficients:
    number_of_random_samples = 2
    bc_matrix,bc_nodal_value_matrix = create_normal_dist_bc_samples(displ_control,
                                                                    numberof_sample=number_of_random_samples,
                                                                    center=0.025,standard_dev=0.025)
    export_dict = {}
    export_dict["bc_matrix"] = bc_matrix
    export_dict["point_bc_settings"] = point_bc_settings
    export_dict["displ_control_settings"] = displ_control_settings
    with open(f'bc_control_dict.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(f'bc_control_dict.pkl', 'rb') as f:
        loaded_control_dict = pickle.load(f)
    
    bc_matrix = loaded_control_dict["bc_matrix"]

# add intended BC to the end of samples
wanted_bc = np.array([-0.0123])
bc_matrix = np.vstack((bc_matrix,wanted_bc))
bc_nodal_value_matrix = displ_control.ComputeBatchControlledVariables(bc_matrix)

# export generated bcs
export_bcs = False
if export_bcs:
    num_unknowns = mechanical_loss_3d.GetNumberOfUnknowns()
    unknown_dofs = jnp.zeros(num_unknowns)
    for i in range(bc_nodal_value_matrix.shape[0]):
        full_displ_vec = mechanical_loss_3d.GetFullDofVector(bc_nodal_value_matrix[i,:],unknown_dofs)
        mdpa_io[f'bc_{i}'] = np.array(full_displ_vec).reshape((fe_model.GetNumberOfNodes(), 3))
        
    mdpa_io.Export(export_dir=case_dir)


# now we need to create, initialize and train fol
fol = FiniteElementOperatorLearning("first_fol",displ_control,[mechanical_loss_3d],[10,10],
                                    "swish",load_NN_params=False,working_directory=working_directory_name)

# cretae FE
fe_solver = FiniteElementSolver("fe_solver",mechanical_loss_3d)

on_the_fly_id = -1
bc_train_mat = bc_matrix[on_the_fly_id].reshape(-1,1).T
bc_train_nodal_value_matrix = bc_nodal_value_matrix[on_the_fly_id]


fol.Initialize()
fol_num_epochs = 1000
fol_batch_size = 1
fol_learning_rate = 0.0001
fol.Train(loss_functions_weights=[1],X_train=bc_train_mat,batch_size=fol_batch_size,
        num_epochs=fol_num_epochs,learning_rate=fol_learning_rate,optimizer="adam",
        convergence_criterion="total_loss",relative_error=1e-20,
        NN_params_save_file_name="NN_params_"+working_directory_name)


num_unknowns = mechanical_loss_3d.GetNumberOfUnknowns()
unknown_dofs = jnp.zeros(num_unknowns)
FOL_UVW = fol.Predict(bc_train_mat)
full_displ_bc_vec = mechanical_loss_3d.GetFullDofVector(bc_train_nodal_value_matrix,unknown_dofs)
mdpa_io['bc'] = np.array(full_displ_bc_vec).reshape((fe_model.GetNumberOfNodes(), 3))
mdpa_io["U_FOL"] = np.array(FOL_UVW).reshape((fe_model.GetNumberOfNodes(), 3))
FE_UVW = np.array(fe_solver.SingleSolve(bc_train_nodal_value_matrix,jnp.zeros(3*fe_model.GetNumberOfNodes())))  
mdpa_io["U_FE"] = np.array(FE_UVW).reshape((fe_model.GetNumberOfNodes(), 3))

mdpa_io.Export(export_dir=case_dir)
