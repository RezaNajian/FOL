
import numpy as np
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.mechanical_3D_fe_tetra import MechanicalLoss3DTetra
from fol.IO.mdpa_io import MdpaIO
from fol.controls.dirichlet_condition_control import DirichletConditionControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle

# directory & save handling
working_directory_name = "results"
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

point_bc_settings = {"Ux":{"support_horizontal_1":0.0},
                     "Uy":{"support_horizontal_1":0.0},
                     "Uz":{"support_horizontal_1":0.0,"main_load_1":0.0}}
mdpa_io = MdpaIO("mdpa_io","hook.mdpa",bc_settings=point_bc_settings,scale_factor=1.0)
model_info = mdpa_io.Import()

# creation of fe model and loss function
fe_model = FiniteElementModel("FE_model",model_info,mdpa_io)

# create loss
mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_loss_3d",fe_model,{"young_modulus":1,"poisson_ratio":0.3},point_bc_settings)

displ_control_settings = {"Uz":["main_load_1"]}
displ_control = DirichletConditionControl("displ_control",displ_control_settings,mechanical_loss_3d)

# create some random bcs 
create_random_coefficients = False
if create_random_coefficients:
    number_of_random_samples = 20
    bc_matrix,bc_nodal_value_matrix = create_normal_dist_bc_samples(displ_control,
                                                                    numberof_sample=number_of_random_samples,
                                                                    center=0.0,standard_dev=1.0)
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
fol = FiniteElementOperatorLearning("first_fol",displ_control,[mechanical_loss_3d],[1],
                                    "tanh",load_NN_params=False,working_directory=working_directory_name)
fol.Initialize()

fol_num_epochs = 20
fol.Train(loss_functions_weights=[1],X_train=bc_matrix,batch_size=1,num_epochs=fol_num_epochs,
            learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
            relative_error=1e-10,NN_params_save_file_name="NN_params_"+working_directory_name)

FOL_UVW = fol.Predict(bc_matrix)
eval_ids = [1,5,10,15]
for eval_id in eval_ids:
    num_unknowns = mechanical_loss_3d.GetNumberOfUnknowns()
    unknown_dofs = jnp.zeros(num_unknowns)
    full_displ_bc_vec = mechanical_loss_3d.GetFullDofVector(bc_nodal_value_matrix[eval_id,:],unknown_dofs)
    mdpa_io[f'bc_{eval_id}'] = np.array(full_displ_bc_vec).reshape((fe_model.GetNumberOfNodes(), 3))
    mdpa_io[f'U_FOL_{eval_id}'] = np.array(FOL_UVW[eval_id]).reshape((fe_model.GetNumberOfNodes(), 3))

mdpa_io.Export(export_dir=case_dir)
