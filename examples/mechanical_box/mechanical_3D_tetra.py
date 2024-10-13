import sys
import os

import numpy as np
from fol.loss_functions.mechanical_3D_fe_tetra import MechanicalLoss3DTetra
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.explicit_parametric_operator_learning import ExplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    
    # directory & save handling
    working_directory_name = "box_3D_tetra"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # create mesh_io
    fe_mesh = Mesh("fol_io","box_3D_coarse.med",'../meshes/')

    # create fe-based loss function
    bc_dict = {"Ux":{"left":0.0},
                "Uy":{"left":0.0,"right":-0.05},
                "Uz":{"left":0.0,"right":-0.05}}
    material_dict = {"young_modulus":1,"poisson_ratio":0.3}
    mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                   "material_dict":material_dict},
                                                                                   fe_mesh=fe_mesh)

    # fourier control
    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([2,4,6]),
                                "beta":20,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

    fe_mesh.Initialize()
    mechanical_loss_3d.Initialize()
    fourier_control.Initialize()

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

    # now save K matrix 
    export_Ks = False
    if export_Ks:
        for i in range(K_matrix.shape[0]):
            fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
        fe_mesh.Finalize(export_dir=case_dir)

    eval_id = 69
    fe_mesh['K'] = np.array(K_matrix[eval_id,:])

    # now we need to create, initialize and train fol

    # A concise MLP defined via lazy submodule initialization

    # from flax.linen import Module, Dense, compact
    from collections.abc import Iterable
    from flax import nnx
    import jax
    import orbax.checkpoint as orbax

    class MLP(nnx.Module):
        def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
            self.dense1 = nnx.Linear(din, dmid, rngs=rngs)
            self.dense2 = nnx.Linear(dmid, dout, rngs=rngs)

        def __call__(self, x: jax.Array) -> jax.Array:
            x = self.dense1(x)
            x = jax.nn.relu(x)
            x = self.dense2(x)
            return x

    my_net = MLP(10, 20, 30, rngs=nnx.Rngs(0))

    # state = nnx.state(my_net)
    # # Save the parameters
    # checkpointer = orbax.StandardCheckpointer()
    # current_directory = os.path.abspath(os.getcwd())
    # full_path = os.path.join(current_directory, "first_flax_orbax")
    # checkpointer.save(full_path, state,force=True)
    # checkpointer.wait_until_finished()

    # new_checkpointer = orbax.StandardCheckpointer()
    # new_state = new_checkpointer.restore(full_path, state)
    # # update the model with the loaded state
    # nnx.update(my_net, new_state)

    # exit()

    import optax
    lr = 1e-3
    sgd_optimizer = optax.sgd(lr, momentum=0.9, nesterov=False)

    fol = ExplicitParametricOperatorLearning(name="dis_fol",control=fourier_control,
                                             loss_function=mechanical_loss_3d,
                                             flax_neural_network=my_net,
                                             optax_optimizer=sgd_optimizer,
                                             checkpoint_settings={"restore_state":False,
                                                                  "state_directory":"./first_flax_orbax"},
                                             working_directory=case_dir)


    fol.Initialize()

    exit()

    # fol = DiscreteOperatorLearning("first_fol",fourier_control,[mechanical_loss_3d],[1],
    #                                 "tanh",load_NN_params=False,working_directory=working_directory_name)
    fol.Initialize()

    fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix[eval_id].reshape(-1,1).T,batch_size=1,num_epochs=fol_num_epochs,
                learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
                relative_error=1e-10,NN_params_save_file_name="NN_params_"+working_directory_name)

    FOL_UVW = np.array(fol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T))
    fe_mesh['U_FOL'] = FOL_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    # solve FE here
    if solve_FE:
        fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"}}
        first_fe_solver = FiniteElementLinearResidualBasedSolver("first_fe_solver",mechanical_loss_3d,fe_setting)
        first_fe_solver.Initialize()
        FE_UVW = np.array(first_fe_solver.Solve(K_matrix[eval_id],jnp.ones(3*fe_mesh.GetNumberOfNodes())))  
        fe_mesh['U_FE'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    fol_num_epochs = 2000
    solve_FE = False
    clean_dir = False

    # Parse the command-line arguments
    args = sys.argv[1:]

    # Process the arguments if provided
    for arg in args:
        if arg.startswith("fol_num_epochs="):
            try:
                fol_num_epochs = int(arg.split("=")[1])
            except ValueError:
                print("fol_num_epochs should be an integer.")
                sys.exit(1)
        elif arg.startswith("solve_FE="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                solve_FE = value.lower() == 'true'
            else:
                print("solve_FE should be True or False.")
                sys.exit(1)
        elif arg.startswith("clean_dir="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                clean_dir = value.lower() == 'true'
            else:
                print("clean_dir should be True or False.")
                sys.exit(1)
        else:
            print("Usage: python thermal_fol.py fol_num_epochs=10 solve_FE=False clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(fol_num_epochs, solve_FE,clean_dir)