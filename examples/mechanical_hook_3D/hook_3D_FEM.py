
import numpy as np
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.mechanical_3D_fe_tetra import MechanicalLoss3DTetra
from fol.IO.mdpa_io import MdpaIO
from fol.controls.fourier_control import FourierControl
from fol.solvers.fe_solver import FiniteElementSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle


# directory & save handling
working_directory_name = "FEM_results"
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

point_bc_settings = {"Ux":{"support_horizontal_1":0.0},
                    "Uy":{"support_horizontal_1":0.0},
                    "Uz":{"support_horizontal_1":0.0,"main_load_1":-1.0}}
mdpa_io = MdpaIO("mdpa_io","hook.mdpa",bc_settings=point_bc_settings,scale_factor=1.0)
model_info = mdpa_io.Import()

# creation of fe model and loss function
fe_model = FiniteElementModel("FE_model",model_info)
mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_loss_3d",fe_model,{"young_modulus":1,"poisson_ratio":0.3})

# fourier control
fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([2,4,6]),
                            "beta":20,"min":1e-1,"max":1}
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

eval_id = -1
mdpa_io['K'] = np.array(K_matrix[eval_id,:])

first_fe_solver = FiniteElementSolver("first_fe_solver",mechanical_loss_3d)
FE_UVW = np.array(first_fe_solver.SingleSolve(K_matrix[eval_id],np.zeros(3*fe_model.GetNumberOfNodes())))  
mdpa_io['U_FE'] = FE_UVW.reshape((fe_model.GetNumberOfNodes(), 3))

mdpa_io.Export(export_dir=case_dir)

