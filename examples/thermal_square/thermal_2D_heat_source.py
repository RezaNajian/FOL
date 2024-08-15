import sys
import os

import numpy as np
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.thermal_2D_fe_quad import ThermalLoss2D
from fol.solvers.fe_solver import FiniteElementSolver
from fol.solvers.nonlinear_solver import NonLinearSolver
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
import pickle, time

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = 'thermal_2D_heat_source'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # problem setup
    model_settings = {"L":1,
                    "N":21,
                    "T_left":1,"T_bottom":1,"T_right":1,"T_top":1}

    # model_settings = {"L":1,
    #                "N":21,
    #                "T_left":1,"T_right":0.1}

    # creation of the model
    model_info = create_2D_square_model_info_thermal_dirichlet(**model_settings)
    # model_info = create_2D_square_model_info_thermal(**model_settings)

    # creation of the objects
    fe_model = FiniteElementModel("FE_model",model_info)
    thermal_loss_2d = ThermalLoss2D("thermal_loss_2d",fe_model,{"num_gp":2})

    # fourier control
    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                                "beta":20,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_model)

    # create some random coefficients & K for training
    create_random_coefficients = True
    if create_random_coefficients:
        number_of_random_samples = 2000
        coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
        export_dict = model_settings.copy()
        export_dict["coeffs_matrix"] = coeffs_matrix
        export_dict["x_freqs"] = fourier_control.x_freqs
        export_dict["y_freqs"] = fourier_control.y_freqs
        export_dict["z_freqs"] = fourier_control.z_freqs
        with open(f'fourier_control_dict_N_{model_settings["N"]}.pkl', 'wb') as f:
            pickle.dump(export_dict,f)
    else:
       with open(f'/workspace/fourier_control_dict_N_{model_settings["N"]}.pkl', 'rb') as f:
           loaded_dict = pickle.load(f)
        
       coeffs_matrix = loaded_dict["coeffs_matrix"]
        # coeffs_/matrix = np.loadtxt('coeffs_matrix.txt')

    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

    # now save K matrix 
    # solution_file = os.path.join(case_dir, "K_matrix.txt")
    # np.savetxt(solution_file,K_matrix[:1000,:])
    # solution_file = os.path.join(case_dir, "coeffs_matrix.txt")
    # np.savetxt(solution_file,coeffs_matrix[:1000,:])

    # specify id of the K of interest
    eval_id = 1
    eval_id2 = 89
    eval_id3 = 29
    eval_id4 = 1989
    eval_id5 = 1765
    train_id = 1000

    # now we need to create, initialize and train fol
    fol = FiniteElementOperatorLearning("first_fol",fourier_control,[thermal_loss_2d],[20,20],
                                        "swish",load_NN_params=False,working_directory=working_directory_name)
    fol.Initialize()

    start_time = time.process_time()
    fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix[:train_id,:],batch_size=10,num_epochs=fol_num_epochs,
                learning_rate=0.0001,optimizer="adam",convergence_criterion="total_loss",relative_error=1e-10,absolute_error=1e-10,
                plot_list=["avg_res","max_res","total_loss"],plot_rate=1,NN_params_save_file_name="NN_params_"+working_directory_name)

    FOL_T = np.array(fol.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T))
    # FOL_T = np.array(fol.Predict(coeffs_matrix[eval_id,:]))

    # solve FE here
    if solve_FE:
        
        first_fe_solver = FiniteElementSolver("first_fe_solver", thermal_loss_2d)
        start_time = time.process_time()
        FE_T = np.array(first_fe_solver.SingleSolve(K_matrix[eval_id],np.zeros(fe_model.GetNumberOfNodes())))  
        print(f"\n############### FE solve took: {time.process_time() - start_time} s ###############\n")

        relative_error = abs(FOL_T.reshape(-1,1)- FE_T.reshape(-1,1))
        plot_mesh_vec_data_paper_temp([K_matrix[eval_id,:], FOL_T, FE_T],f'sample_{eval_id}')
        
        plot_mesh_vec_data(model_settings["L"], [K_matrix[eval_id,:],FOL_T,FE_T,relative_error], 
                        subplot_titles= ['Heterogeneity', 'FOL_T', 'FE_T', "absolute_error"], fig_title=None, cmap='viridis',
                            block_bool=True, colour_bar=True, colour_bar_name=None,
                            X_axis_name=None, Y_axis_name=None, show=False, file_name=os.path.join(case_dir,'plot_results.png'))

        eval_list = [eval_id2,eval_id3,eval_id4,eval_id5]
        for i,eval_id in enumerate(eval_list):
            FOL_T = np.array(fol.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T))
            # FOL_T = np.array(fol.Predict(coeffs_matrix[eval_id,:]))
            print(f'eval coeffs: {coeffs_matrix[eval_id,:]}')
            print(f"predicted array: {FOL_T}")
            start_time = time.process_time()
            FE_T = np.array(first_fe_solver.SingleSolve(K_matrix[eval_id],np.zeros(fe_model.GetNumberOfNodes())))  
            print(f"\n############### FE solve took: {time.process_time() - start_time} s ###############\n")
            plot_mesh_vec_data_paper_temp([K_matrix[eval_id,:], FOL_T, FE_T],f'sample_{eval_id}')

            relative_error = abs(FOL_T.reshape(-1,1)- FE_T.reshape(-1,1))
            plot_mesh_vec_data(model_settings["L"], [K_matrix[eval_id,:],FOL_T,FE_T,relative_error], 
                        subplot_titles= ['Heterogeneity', 'FOL_T', 'FE_T', "absolute_error"], fig_title=None, cmap='viridis',
                            block_bool=True, colour_bar=True, colour_bar_name=None,
                            X_axis_name=None, Y_axis_name=None, show=False, file_name=os.path.join(case_dir,'plot_results.png'))


        # now we need to create, initialize and train fol
    # fol = FiniteElementOperatorLearning("first_fol",fourier_control,[thermal_loss_2d],[20,20],
    #                                     "swish",load_NN_params=False,working_directory=working_directory_name)
    # fol.Initialize()

    # start_time = time.process_time()
    # fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix[:train_id,:],batch_size=10,num_epochs=fol_num_epochs,
    #             learning_rate=0.0001,optimizer="adam",convergence_criterion="total_loss",relative_error=1e-10,absolute_error=1e-10,
    #             plot_list=["avg_res","max_res","total_loss"],plot_rate=1,NN_params_save_file_name="NN_params_"+working_directory_name)

    # FOL_T = np.array(fol.Predict(coeffs_matrix[eval_id2,:].reshape(-1,1).T))
    # # solve FE here
    # if solve_FE:
        
    #     first_fe_solver = FiniteElementSolver("first_fe_solver", thermal_loss_2d)
    #     start_time = time.process_time()
    #     FE_T = np.array(first_fe_solver.SingleSolve(K_matrix[eval_id2],np.zeros(fe_model.GetNumberOfNodes())))  
    #     print(f"\n############### FE solve took: {time.process_time() - start_time} s ###############\n")

    #     relative_error = abs(FOL_T.reshape(-1,1)- FE_T.reshape(-1,1))
    #     plot_mesh_vec_data_paper_temp([K_matrix[eval_id2,:], FOL_T, FE_T],f'sample_{eval_id2}')


    #     fol = FiniteElementOperatorLearning("first_fol",fourier_control,[thermal_loss_2d],[20,20],
    #                                     "swish",load_NN_params=False,working_directory=working_directory_name)
    # fol.Initialize()

    # start_time = time.process_time()
    # fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix[:train_id,:],batch_size=10,num_epochs=fol_num_epochs,
    #             learning_rate=0.0001,optimizer="adam",convergence_criterion="total_loss",relative_error=1e-10,absolute_error=1e-10,
    #             plot_list=["avg_res","max_res","total_loss"],plot_rate=1,NN_params_save_file_name="NN_params_"+working_directory_name)

    # FOL_T = np.array(fol.Predict(coeffs_matrix[eval_id4,:].reshape(-1,1).T))
    # # solve FE here
    # if solve_FE:
        
    #     first_fe_solver = FiniteElementSolver("first_fe_solver", thermal_loss_2d)
    #     start_time = time.process_time()
    #     FE_T = np.array(first_fe_solver.SingleSolve(K_matrix[eval_id4],np.zeros(fe_model.GetNumberOfNodes())))  
    #     print(f"\n############### FE solve took: {time.process_time() - start_time} s ###############\n")

    #     relative_error = abs(FOL_T.reshape(-1,1)- FE_T.reshape(-1,1))
    #     plot_mesh_vec_data_paper_temp([K_matrix[eval_id4,:], FOL_T, FE_T],f'sample_{eval_id4}')
    
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
            print("Usage: python mechanical_2D.py fol_num_epochs=10 solve_FE=False clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(fol_num_epochs, solve_FE,clean_dir)
