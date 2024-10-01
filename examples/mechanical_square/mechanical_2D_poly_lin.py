import sys
import os

import numpy as np
from fol.loss_functions.mechanical_2D_fe_quad import MechanicalLoss2D
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.controls.voronoi_control import VoronoiControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
import pickle, time

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = 'mechanical_2D_poly_lin'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))
    
    # problem setup
    model_settings = {"L":1,"N":20,
                      "Ux_left":0.0,
                      "Ux_right":0.05,
                      "Uy_left":0.0,
                      "Uy_right":0.05}

    # creation of the model
    fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

    # create fe-based loss function
    bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
               "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}
    
    material_dict = {"young_modulus":1,"poisson_ratio":0.3}
    mechanical_loss_2d = MechanicalLoss2D("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                              "num_gp":2,
                                                                              "material_dict":material_dict},
                                                                              fe_mesh=fe_mesh)    

    # k_rangeof_values in the following could be a certain amount of values from a list instead of a tuple
    # voronoi_control_settings = {"numberof_seeds":5,"k_rangeof_values":list(0.01*np.arange(10,100,10))}
    voronoi_control_settings = {"numberof_seeds":5,"k_rangeof_values":(0.1,1)}
    voronoi_control = VoronoiControl("first_voronoi_control",voronoi_control_settings,fe_mesh)

    fe_mesh.Initialize()
    mechanical_loss_2d.Initialize()
    voronoi_control.Initialize()

    # create some random coefficients & K for training
    create_random_coefficients = False
    if create_random_coefficients:
        number_of_random_samples = 100
        coeffs_matrix,K_matrix = create_random_voronoi_samples(voronoi_control,number_of_random_samples)
        export_dict = {}
        export_dict["coeffs_matrix"] = coeffs_matrix
        with open(f'voronoi_control_dict.pkl', 'wb') as f:
            pickle.dump(export_dict,f)
    else:
        with open(f'voronoi_control_dict.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        
        coeffs_matrix = loaded_dict["coeffs_matrix"]
    
    K_matrix = voronoi_control.ComputeBatchControlledVariables(coeffs_matrix)

    export_domain = False
    domain_export_dict = {}
    domain_export_dict["K_matrix"] = K_matrix
    if export_domain:
        with open(f'voronoi_domain_dict.pkl', 'wb') as f:
            pickle.dump(domain_export_dict,f)

    # set NN hyper-parameters
    fol_num_epochs = 1000
    fol_batch_size = 1
    fol_learning_rate = 0.0001
    hidden_layer = [1]
    # here we specify whther to do pr_le or on the fly solve
    parametric_learning = True
    if parametric_learning:
        # now create train and test samples
        num_train_samples = int(0.8 * coeffs_matrix.shape[0])
        pc_train_mat = coeffs_matrix[0:num_train_samples]
        pc_train_nodal_value_matrix = K_matrix[0:num_train_samples]
        pc_test_mat = coeffs_matrix[num_train_samples:]
        pc_test_nodal_value_matrix = K_matrix[num_train_samples:]
    else:
        on_the_fly_id = -1
        pc_train_mat = coeffs_matrix[on_the_fly_id].reshape(-1,1).T
        pc_train_nodal_value_matrix = K_matrix[on_the_fly_id]

    linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",mechanical_loss_2d)
    linear_fe_solver.Initialize()

    # now we need to create, initialize and train fol
    fol = FiniteElementOperatorLearning("first_fol",voronoi_control,[mechanical_loss_2d],hidden_layer,
                                        "swish",load_NN_params=False,working_directory=working_directory_name)
    fol.Initialize()
    fol.Train(loss_functions_weights=[1],X_train=pc_train_mat,batch_size=fol_batch_size,num_epochs=fol_num_epochs,
                learning_rate=fol_learning_rate,optimizer="adam",convergence_criterion="total_loss",absolute_error=1e-15,
                relative_error=1e-15,NN_params_save_file_name="NN_params_"+working_directory_name)

    if parametric_learning:
        UV_train = fol.Predict(pc_train_mat)
        UV_test = fol.Predict(pc_test_mat)

        test_eval_ids = [0,1]
        for eval_id in test_eval_ids:
            FOL_UV = UV_test[eval_id]
            FE_UV = np.array(linear_fe_solver.Solve(pc_test_nodal_value_matrix[eval_id],np.zeros(2*fe_mesh.GetNumberOfNodes())))  
            absolute_error = abs(FOL_UV.reshape(-1,1)- FE_UV.reshape(-1,1))
            
            # Plot displacement field U
            vectors_list = [pc_test_nodal_value_matrix[eval_id],FOL_UV[::2],FE_UV[::2],absolute_error[::2]]
            plot_mesh_vec_data_paper_temp(vectors_list,f"U_test_sample_{eval_id}","U")
            # Plot displacement field V
            vectors_list = [pc_test_nodal_value_matrix[eval_id],FOL_UV[1::2],FE_UV[1::2],absolute_error[1::2]]
            plot_mesh_vec_data_paper_temp(vectors_list,f"V_test_sample_{eval_id}","V")
            # Plot Stress field
            plot_mesh_vec_grad_data_mechanics(vectors_list, f"stress_test_sample_{eval_id}", material_dict)


        train_eval_ids = [0,1]
        for eval_id in train_eval_ids:
            FOL_UV = UV_train[eval_id]
            FE_UV = np.array(linear_fe_solver.Solve(pc_train_nodal_value_matrix[eval_id],np.zeros(2*fe_mesh.GetNumberOfNodes())))
            absolute_error = abs(FOL_UV.reshape(-1,1)- FE_UV.reshape(-1,1))
            
            # Plot displacement field U
            vectors_list = [pc_train_nodal_value_matrix[eval_id],FOL_UV[::2],FE_UV[::2],absolute_error[::2]]
            plot_mesh_vec_data_paper_temp(vectors_list,f"U_train_sample_{eval_id}","U")
            # Plot displacement field V
            vectors_list = [pc_train_nodal_value_matrix[eval_id],FOL_UV[1::2],FE_UV[1::2],absolute_error[1::2]]
            plot_mesh_vec_data_paper_temp(vectors_list,f"V_train_sample_{eval_id}","V")
            # Plot Stress field
            plot_mesh_vec_grad_data_mechanics(vectors_list,f"stress_train_sample_{eval_id}", material_dict)

    else:
        FOL_UV = fol.Predict(pc_train_mat)
        FE_UV = np.array(linear_fe_solver.Solve(pc_train_nodal_value_matrix,np.zeros(2*fe_mesh.GetNumberOfNodes())))
        absolute_error = abs(FOL_UV.reshape(-1,1)- FE_UV.reshape(-1,1))
        
        # Plot displacement field U
        vectors_list = [pc_train_nodal_value_matrix,FOL_UV[::2],FE_UV[::2],absolute_error[::2]]
        plot_mesh_vec_data_paper_temp(vectors_list,"U_train","U")
        # Plot displacement field V
        vectors_list = [pc_train_nodal_value_matrix,FOL_UV[1::2],FE_UV[1::2],absolute_error[1::2]]
        plot_mesh_vec_data_paper_temp(vectors_list,"V_train","V")
        # Plot Stress field
        plot_mesh_vec_grad_data_mechanics(vectors_list,"stress_train", material_dict)

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    fol_num_epochs = 2000
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
            print("Usage: python mechanical_2D_poly.py fol_num_epochs=10 solve_FE=False clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(fol_num_epochs, solve_FE,clean_dir)