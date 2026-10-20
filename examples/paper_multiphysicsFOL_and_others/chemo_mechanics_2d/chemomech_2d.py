import sys
import os
import optax
import numpy as np
from fol.loss_functions.chemo_mechanics import ChemoMechanicsLoss2DQuad
from fol.controls.fourier_control import FourierControl
from fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.controls.identity_control import IdentityControl
from cm_useful_functions import *
from fol.tools.logging_functions import Logger

from fol.deep_neural_networks.fourier_parametric_operator_learning import PhysicsInformedFourierParametricOperatorLearning
from fourier_neural_operator_networks import FourierNeuralOperator2D

from fol.tools.decoration_functions import *
from flax import nnx
from flax.nnx import bridge
import jax
import pyvista as pv

jax.config.update('jax_default_matmul_precision','high')
# jax.config.update('jax_enable_x64', True)

# directory & save handling
working_directory_name = 'chemomech_2D'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":42,
                  "Ux_left":0.0,"Ux_right":0.10,
                  "Uy_left":0.0,"Uy_right":0.10,
                  "C_left":0.5,"C_right":0.0}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh.Initialize()
identity_control = IdentityControl("K", {}, fe_mesh.GetNumberOfNodes())
# create fe-based loss function
bc_dict = {"C":{"left":model_settings["C_left"],"right":model_settings["C_right"]},
           "Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
           "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}#

freq_sets = [
    (np.array([2, 4, 6]),    np.array([2, 4, 6])),
    (np.array([1, 2, 3]),    np.array([1, 2, 3])),
    (np.array([3, 4, 5]),    np.array([3, 4, 5])),
    (np.array([4, 6, 8]),    np.array([4, 6, 8]))
]

N_samples_per_set = 1500

all_K_list = []
create_random_coefficients = True
if create_random_coefficients:
    for idx, (x_freqs, y_freqs) in enumerate(freq_sets):
        control_settings = {
            "x_freqs": x_freqs,
            "y_freqs": y_freqs,
            "z_freqs": np.array([0]),
            "beta": 10,
            "min": 0.1,
            "max": 1.0
        }

        fourier_control = FourierControl("K", control_settings, fe_mesh)
        fourier_control.Initialize()

        coeffs_matrix, K_matrix = create_random_fourier_samples(fourier_control, N_samples_per_set)
        np.save(f"coeffs_matrix_{idx}.npy", coeffs_matrix)
        all_K_list.append(K_matrix)
    K_matrix = np.vstack(all_K_list)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(K_matrix)

else:
    for idx, (x_freqs, y_freqs) in enumerate(freq_sets):
        control_settings = {
            "x_freqs": x_freqs,
            "y_freqs": y_freqs,
            "z_freqs": np.array([0]),
            "beta": 10,
            "min": 0.1,
            "max": 1.0
        }
        coeffs_matrix = np.load(f"coeffs_matrix_{idx}.npy")
        fourier_control = FourierControl("K", control_settings, fe_mesh)
        fourier_control.Initialize()
        K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)
        all_K_list.append(K_matrix)
    K_matrix = np.vstack(all_K_list)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(K_matrix)

# Initial temperature field (stored per node)
initial_temp = jnp.full((1, fe_mesh.GetNumberOfNodes()), 0.1)

# Material parameters
material_dict = {
    "young_modulus": 1.0,
    "poisson_ratio": 0.3,
    "c0": initial_temp.flatten(),
    "Diffusivity": 1.0,
    "beta":0.1,
    "gamma":0.01
}

chemomech_loss_2d = ChemoMechanicsLoss2DQuad("chemomech_loss_2d",
                                               loss_settings={
                                                    "dirichlet_bc_dict": bc_dict,
                                                    "material_dict": material_dict,
                                                },
                                                fe_mesh=fe_mesh)
chemomech_loss_2d.Initialize()

def merge_state(dst: nnx.State, src: nnx.State):
    for key, value in src.items():
        if isinstance(value, nnx.State):
            merge_state(dst[key], value)
        else:
            dst[key] = value


fno_model = bridge.ToNNX(
    FourierNeuralOperator2D(
        modes1=12,
        modes2=12,
        width=34,
        depth=4,
        channels_last_proj=128,
        out_channels=3,
        padding=4,
        output_scale=0.001,
    ),
    rngs=nnx.Rngs(0),
).lazy_init(
    K_matrix[0:1].reshape(1, model_settings["N"], model_settings["N"], 1)
)

# Store RNG keys as arrays so Orbax can restore the final checkpoint reliably.
graph_def, state = nnx.split(fno_model)
rngs_key = jax.tree.map(jax.random.key_data, state.filter(nnx.RngKey))
merge_state(state, rngs_key)
fno_model = nnx.merge(graph_def, state)

# get total number of fno params
params = nnx.state(fno_model, nnx.Param)
total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
print(f"total number of fno network param:{total_params}")

num_epochs = 1000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-2, end_value=1e-3, transition_steps=num_epochs)
optimizer = optax.chain(optax.adam(learning_rate_scheduler))

# create fol
pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning(name="pi_fno_pr_learning",
                                                                        control=identity_control,
                                                                        loss_function=chemomech_loss_2d,
                                                                        flax_neural_network=fno_model,
                                                                        optax_optimizer=optimizer)

pi_fno_pr_learning.Initialize()

train_start_id = 0
train_end_id = 5000
test_start_id = 5000
test_end_id = 6000

pi_fno_pr_learning.Train(train_set=(K_matrix[train_start_id:train_end_id,:],),
          test_set=(K_matrix[test_start_id:test_end_id,:],),
          batch_size=100,
          convergence_settings={"num_epochs":num_epochs,
                                "relative_error":1e-8,
                                "absolute_error":1e-8,
                                "staggered":False},
          train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10},
          working_directory=case_dir,
          plot_settings={"plot_list":["total_loss"],
                         "plot_frequency":1,"save_frequency":100,
                         "save_directory":case_dir})

pi_fno_pr_learning.RestoreState("chemomech_2D/flax_final_state")


# ---- config ----
start_id = 5000
num_samples = 1
sample_ids = range(start_id, start_id + num_samples)

n_nodes = fe_mesh.GetNumberOfNodes()
output_size = 3  # C, Ux, Uy

# initialize FE solver once
fe_setting = {
    "linear_solver_settings":{
        "solver":"JAX-direct","tol":1e-6,"atol":1e-6,
        "maxiter":1000,"pre-conditioner":"ilu"},
    "nonlinear_solver_settings":{
        "rel_tol":1e-6,"abs_tol":1e-6,"maxiter":25,"load_incr":20,
        "line_search":True,"line_search_step_size":0.25,
        "line_search_reduction_factor":0.5,
        "line_search_min_step_size":1e-4,
        "line_search_maxiter":40}
}
nonlinear_fe_solver = FiniteElementNonLinearResidualBasedSolver(
    "nonlinear_fe_solver", chemomech_loss_2d, fe_setting
)
nonlinear_fe_solver.Initialize()

# base directory to hold all samples (e.g., ".../case_batch")
batch_dir = os.path.join(case_dir, f"batch_{start_id}_{num_samples}")
os.makedirs(batch_dir, exist_ok=True)

for sid in sample_ids:
    print(f"Processing sample {sid}...")
    sample_dir = os.path.join(batch_dir, f"sample_{sid:05d}")
    os.makedirs(sample_dir, exist_ok=True)

    K_vec = np.array(K_matrix[sid, :])
    FOL_CUV = np.array(
        pi_fno_pr_learning.Predict(K_vec[None, :])
    ).reshape(n_nodes, output_size)
    FE_CUV_flat = np.array(
        nonlinear_fe_solver.Solve(K_vec.flatten(), np.zeros(n_nodes * output_size))
    )
    FE_CUV = np.reshape(FE_CUV_flat, (n_nodes, output_size))

    pv_mesh = pv.wrap(fe_mesh.mesh_io).copy(deep=True)
    pv_mesh.point_data['FOL_C'] = FOL_CUV[:, 0]
    pv_mesh.point_data['FOL_Ux'] = FOL_CUV[:, 1]
    pv_mesh.point_data['FOL_Uy'] = FOL_CUV[:, 2]
    pv_mesh.point_data['FE_C'] = FE_CUV[:, 0]
    pv_mesh.point_data['FE_Ux'] = FE_CUV[:, 1]
    pv_mesh.point_data['FE_Uy'] = FE_CUV[:, 2]
    pv_mesh.point_data['ABS_ERROR_C'] = np.abs(FOL_CUV[:, 0] - FE_CUV[:, 0])
    pv_mesh.point_data['ABS_ERROR_Ux'] = np.abs(FOL_CUV[:, 1] - FE_CUV[:, 1])
    pv_mesh.point_data['ABS_ERROR_Uy'] = np.abs(FOL_CUV[:, 2] - FE_CUV[:, 2])

    plot_fields_matplotlib(
        pv_mesh,
        [
            'FOL_C', 'FOL_Ux', 'FOL_Uy',
            'FE_C', 'FE_Ux', 'FE_Uy',
            'ABS_ERROR_C', 'ABS_ERROR_Ux', 'ABS_ERROR_Uy',
        ],
        nrows=3,
        ncols=3,
        save_path=os.path.join(sample_dir, f"primary_fields_{sid}.png"),
        figsize=(14, 12),
        shading="gouraud",
    )

    print(f"Saved primary-field contours to: {sample_dir}")

print("Done.")
