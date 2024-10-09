import os,time,sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.insert(0, parent_dir)
import numpy as np
from jax import vmap
from fol.mesh_input_output.mesh import Mesh
from fol.loss_functions.thermal_transient_2D_fe_quad import ThermalTransientLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.controls.no_control import NoControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):

    # cleaning & logging
    working_directory_name = 'results'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,"terminal.log"))

    # creation of the model
    fe_mesh = create_2D_square_mesh(L=1,N=21)
    no_control = NoControl("no_control",fe_mesh)

    # create fe-based loss function
    bc_dict = {"T":{"left":1.0,"right":0.1}}

    thermal_loss_2d = ThermalTransientLoss2DQuad("thermal_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                          "compute_dims":2,"rho":1.0,"cp":10.0, "dt":0.05},
                                                                            fe_mesh=fe_mesh)
    
    fe_mesh.Initialize()
    thermal_loss_2d.Initialize()
    no_control.Initialize()    

    create_random_samples = False
    if create_random_samples:
        pass
    else:
        T_matrix = np.load('gaussian_kernel_50000_N21.npy')
    # now save K matrix 
    export_Ks = False
    if export_Ks:
        for i in range(T_matrix.shape[0]):
            fe_mesh[f'K_{i}'] = np.array(T_matrix[i,:])
        fe_mesh.Finalize(export_dir=case_dir)

    eval_id = 1
    fe_mesh['T_data'] = np.array(T_matrix[eval_id])

    # now we need to create, initialize and train fol
    fol = FiniteElementOperatorLearning("fol",no_control,[thermal_loss_2d],[1000,1000],
                                        "swish",load_NN_params=False,working_directory=working_directory_name)
    fol.Initialize()
    fol.Train(loss_functions_weights=[1],X_train=T_matrix,batch_size=500,num_epochs=fol_num_epochs,
                learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
                relative_error=1e-15,NN_params_save_file_name="NN_params_"+working_directory_name)
    
    num_steps = 50
    FOL_T_temp = T_matrix[eval_id]
    FOL_T = np.zeros((fe_mesh.GetNumberOfNodes(),num_steps))
    for i in range(num_steps):
        FOL_T_temp = np.array(fol.Predict(FOL_T_temp.reshape(-1,1).T))
        FOL_T[:,i] = FOL_T_temp
    
    fe_mesh['T_FOL'] = FOL_T

    # solve FE here
    if solve_FE: 
        fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"},
                      "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                    "maxiter":20,"load_incr":1}}
        nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",thermal_loss_2d,fe_setting)
        nonlin_fe_solver.Initialize()

        FE_T_temp = T_matrix[eval_id]
        FE_T = np.zeros((fe_mesh.GetNumberOfNodes(),num_steps))
        for i in range(num_steps):
            FE_T_temp = np.array(nonlin_fe_solver.Solve(FE_T_temp,np.zeros(fe_mesh.GetNumberOfNodes())))  
            FE_T[:,i] = FE_T_temp    
        fe_mesh['T_FE'] = FE_T#.reshape((fe_mesh.GetNumberOfNodes(), 1))

        relative_error = 100 * (abs(FOL_T-FE_T))/abs(FE_T)
        fe_mesh['relative_error'] = relative_error

    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    fol_num_epochs = 500
    solve_FE = True
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