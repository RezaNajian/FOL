import sys
import os
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import optax
import numpy as np
from flax import nnx
from fol.loss_functions.transient_thermal import TransientThermalLoss2DQuad
from fol.controls.fourier_control import FourierControl
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.ported_fourier_neural_operator_networks.fno import FNO
from fol.deep_neural_networks.fourier_parametric_operator_learning import PhysicsInformedFourierParametricOperatorLearning
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.tools.decoration_functions import *
import pickle
import jax
from thermal_usefull_functions import *

# jax.config.update('jax_default_matmul_precision','high')
# jax.config.update('jax_enable_x64', True)
train_from_scratch = True
# directory & save handling
working_directory_name = 'pi_fno_thermal_2D'
case_dir = os.path.join('.', working_directory_name)
if train_from_scratch:
    create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":42,
                    "T_left":0.0,"T_right":1.0}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh.Initialize()

# create fourier-based conductivity fields
fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                                "beta":20,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)
fourier_control.Initialize()

number_of_random_samples = 1000
# create some random conductivities
if train_from_scratch:
    
    coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
    export_dict = model_settings.copy()
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["x_freqs"] = fourier_control.x_freqs
    export_dict["y_freqs"] = fourier_control.y_freqs
    export_dict["z_freqs"] = fourier_control.z_freqs
    with open(f'fourier_control_conductivity_field_dict.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(f'fourier_control_conductivity_field_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]

train_heterogeneity_fields = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

# create some random fields for training
if train_from_scratch:
    train_temperature_fields = generate_random_smooth_patterns(model_settings["L"],model_settings["N"],num_samples=number_of_random_samples) 
    train_data_dict = {"temperatures":train_temperature_fields, "heterogeneity":train_heterogeneity_fields}
    with open(f'train_data_dict.pkl', 'wb') as f:
        pickle.dump(train_data_dict,f)
else:
    with open(f'train_data_dict.pkl', 'rb') as f:
        train_data_dict = pickle.load(f)
    
    train_temperature_fields = train_data_dict["temperatures"]
    train_heterogeneity_fields = train_data_dict["heterogeneity"]

    fol_info(f"train temperature field {train_temperature_fields.shape} is imported !")

# make consistent the number of samples for both fields
if train_temperature_fields.shape != train_heterogeneity_fields.shape:
    min_samples = min(train_heterogeneity_fields.shape[0],train_temperature_fields.shape[0])
    train_heterogeneity_fields = train_heterogeneity_fields[:min_samples]
    train_temperature_fields = train_temperature_fields[:min_samples]
    
train_fields = np.stack([train_heterogeneity_fields,train_temperature_fields],axis=-1)

# create fe-based loss function
bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]}}
material_dict = {"rho":1.0,"cp":1.0,"k0":train_heterogeneity_fields[0,:],"beta":1.5,"c":1.0}
time_integration_dict = {"method":"implicit-euler","time_step":0.005}
transient_thermal_loss_2d = TransientThermalLoss2DQuad("thermal_transient_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict,
                                                                            "time_integration_dict":time_integration_dict},
                                                                            fe_mesh=fe_mesh)
transient_thermal_loss_2d.Initialize()

identity_control = IdentityControl("ident_control",control_settings={},num_vars=fe_mesh.GetNumberOfNodes())
identity_control.Initialize()

# ----------------------------------------------------------------------------
# Build the FNO (Fourier Neural Operator) in JAX/Flax 
# ----------------------------------------------------------------------------
fno_dict = {"in_channel": 2, "out_channel":1,
            "hidden_channels":32,"n_modes":(12,12),"n_layers":3,
            "lifting_channel_ratio":2, "projection_channel_ratio":2,}

fno_model = FNO(
    in_channels=fno_dict["in_channel"],
    out_channels=fno_dict["out_channel"],
    hidden_channels=fno_dict["hidden_channels"],
    n_modes=fno_dict["n_modes"],
    n_layers=fno_dict["n_layers"],
    lifting_channel_ratio=fno_dict["lifting_channel_ratio"],
    projection_channel_ratio=fno_dict["projection_channel_ratio"],
    rngs=nnx.Rngs(0)
)

# Count trainable parameters 
params = nnx.state(fno_model, nnx.Param)
total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
print(f"FNO trainable parameters:{total_params}")

# Sanity-check forward pass with a small batch:
# FNO expects shape (batch, Nx, Ny, channels)
# ISince K_matrix is stored as flattened vectors, we reshape to (B,N,N,1)
init_out = fno_model(train_fields[0:8].reshape(8,model_settings["N"],model_settings["N"],fno_dict["in_channel"]))


num_epochs = 100
learning_rate_scheduler = optax.linear_schedule(init_value=1e-3, end_value=1e-5, transition_steps=num_epochs)
optimizer = optax.chain(optax.adam(learning_rate_scheduler))

# create fol
pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning(name="pi_fno_pr_learning",
                                                                control=identity_control,
                                                                loss_function=transient_thermal_loss_2d,
                                                                flax_neural_network=fno_model,
                                                                optax_optimizer=optimizer)

pi_fno_pr_learning.Initialize()

train_start_id = 0
train_end_id = 500
assert train_end_id <= number_of_random_samples, f"train_end_id should be less than {number_of_random_samples}!"

if train_from_scratch:
    pi_fno_pr_learning.Train(train_set=(train_fields[train_start_id:train_end_id,:],),
                            batch_size=8,
                            convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
                            plot_settings={"plot_save_rate":100},
                            train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
                            working_directory=case_dir)

else:
    # load the best model
    pi_fno_pr_learning.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

# time ineference setup
initial_solution_id = 0 
initial_solution = train_fields[train_end_id + initial_solution_id,:]   # shape: (:,2)
num_time_steps = 50


# update heterogeneity
material_dict = {"rho":1.0,"cp":1.0,"k0":train_heterogeneity_fields[initial_solution_id,:],"beta":1.5,"c":1.0}
transient_thermal_loss_2d.Initialize()

# predict dynamics with FE
fe_setting = {"linear_solver_settings":{"solver":"JAX-direct","tol":1e-6,"atol":1e-6,
                                                "maxiter":1000,"pre-conditioner":"ilu"},
                "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                            "maxiter":10,"load_incr":1}}
nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",transient_thermal_loss_2d,fe_setting)
nonlin_fe_solver.Initialize()

T_fe_current = initial_solution[:,1]
T_fes = jnp.array(T_fe_current)
T_ifol_current = initial_solution[:,1]
T_ifols = jnp.array(T_ifol_current)
heterogeneity = initial_solution[:,0]

for _ in range(0,num_time_steps):
    T_fe_next = np.array(nonlin_fe_solver.Solve(T_fe_current,T_fe_current))
    T_fe_current = T_fe_next
    T_fes = jnp.vstack((T_fes,T_fe_next.flatten()))
    
    # predict dynamics with ifol
    current_field = np.stack([heterogeneity,T_ifol_current],axis=-1)
    T_ifol_next = pi_fno_pr_learning.Predict(current_field[None,:,:])
    T_ifol_current = T_ifol_next.flatten()
    T_ifols = jnp.vstack((T_ifols,T_ifol_next.flatten()))

# save the solutions
for time_index, (T_ifol,T_fe) in enumerate(zip(T_ifols,T_fes)):
    fe_mesh[f"T_ifol_{time_index}"] = np.array(T_ifol).reshape((fe_mesh.GetNumberOfNodes(), 1))
    fe_mesh[f"T_fe_{time_index}"] = np.array(T_fe).reshape((fe_mesh.GetNumberOfNodes(), 1))

fe_mesh["k0"] = np.array(heterogeneity.flatten()).reshape((fe_mesh.GetNumberOfNodes(), 1))


absolute_error = np.abs(T_ifols- T_fes)
time_list = [0,1,4,9,19,24,49]

plot_mesh_vec_data_thermal_row(1,[T_fe_current],
                   [""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"initial_condition.png"))

plot_mesh_quad(fe_mesh.GetNodesCoordinates()[:,:-1],
               fe_mesh.GetElementsNodes("quad"),
               background=heterogeneity.reshape((model_settings["N"],model_settings["N"]))[::-1], 
               filename=os.path.join(case_dir,"FE_mesh_hetero_info.png"),show=False)

plot_mesh_vec_data_thermal_row(1,[T_ifols[time_list[0],:],T_ifols[time_list[1],:],
                                  T_ifols[time_list[2],:],T_ifols[time_list[3],:],
                                  T_ifols[time_list[4],:],T_ifols[time_list[5],:],
                                  T_ifols[time_list[6],:]],
                   ["","","","","","",""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"test_FOL_summary.png"))
plot_mesh_vec_data_thermal_row(1,[T_fes[time_list[0],:],T_fes[time_list[1],:],
                                  T_fes[time_list[2],:],T_fes[time_list[3],:],
                                  T_fes[time_list[4],:],T_fes[time_list[5],:],
                                  T_fes[time_list[6],:]],
                   ["","","","","","",""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"test_FE_summary.png"))

plot_mesh_vec_data_thermal_row(1,[absolute_error[time_list[0],:],absolute_error[time_list[1],:],
                                  absolute_error[time_list[2],:],absolute_error[time_list[3],:],
                                  absolute_error[time_list[4],:],absolute_error[time_list[5],:],
                                  absolute_error[time_list[6],:]],
                   ["","","","","","",""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"test_Error_summary.png"))


fe_mesh.Finalize(export_dir=case_dir)
