import sys
import os

import numpy as np
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.thermal_transient_2D_fe_quad import ThermalLoss2D
from fol.solvers.fe_solver import FiniteElementSolver
from fol.solvers.nonlinear_solver import NonLinearSolver
# from fol.controls.fourier_control import FourierControl
from fol.controls.no_control import NoControl
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
import pickle, time

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = 'thermal_2D_heat_transient'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # problem setup
    model_settings = {"L":1,
                    "N":21,
                    "T_left":1.0,"T_bottom":1.0,"T_right":0.0,"T_top":0.0}

    # model_settings = {"L":1,
    #                "N":21,
    #                "T_left":1,"T_right":0.1}

    # creation of the model 
    model_info = create_2D_square_model_info_thermal_dirichlet(**model_settings)
    # model_info = create_2D_square_model_info_thermal(**model_settings)

    # creation of the objects
    fe_model = FiniteElementModel("FE_model",model_info)
    thermal_loss_2d = ThermalLoss2D("thermal_loss_2d",fe_model,{"num_gp":2, "rho":1.0, "cp":10.0, "dt":0.05})

    # fourier control
    # fourier_control_settings = {"x_freqs":np.array([1,2,3]),"y_freqs":np.array([1,2,3]),"z_freqs":np.array([0]),
    #                             "beta":10,"min":1e-1,"max":1}
    # fourier_control = Control("fourier_control",fourier_control_settings,fe_model)
    no_control = NoControl("no_control",fe_model)

    # create some random coefficients & K for training
    create_random_coefficients = False
    if create_random_coefficients:
        pass
        # number_of_random_samples = 2000
        # coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
        # export_dict = model_settings.copy()
        # export_dict["coeffs_matrix"] = coeffs_matrix
        # export_dict["x_freqs"] = fourier_control.x_freqs
        # export_dict["y_freqs"] = fourier_control.y_freqs
        # export_dict["z_freqs"] = fourier_control.z_freqs
        # with open(f'fourier_control_dict_N_{model_settings["N"]}.pkl', 'wb') as f:
        #     pickle.dump(export_dict,f)
    else:
    #    with open(f'fourier_control_dict_N_{model_settings["N"]}.pkl', 'rb') as f:
    #        loaded_dict = pickle.load(f)    
    #    coeffs_matrix = loaded_dict["coeffs_matrix"]
        coeffs_matrix = np.load('gaussian_kernel_50000_N21.npy')

    Ts_c = no_control.ComputeBatchControlledVariables(coeffs_matrix)

    # specify id of the K of interest
    eval_id = 1
    eval_id2 = 289
    eval_id3 = 990
    eval_id4 = 689
    eval_id5 = 367
    eval_id6 = 1989
    eval_id7 = 1367
    eval_id8 = 1641
    eval_id9 = 1893
    train_id = 10000

    # now we need to create, initialize and train fol
    fol = FiniteElementOperatorLearning("first_fol",no_control,[thermal_loss_2d],[1000,1000],
                                        "swish",load_NN_params=False,working_directory=working_directory_name)
    fol.Initialize()

    start_time = time.process_time()
    fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix[:train_id,:],batch_size=100,num_epochs=fol_num_epochs,
                learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",relative_error=1e-10,absolute_error=1e-10,
                plot_list=["avg_res","max_res","total_loss"],plot_rate=1,NN_params_save_file_name="NN_params_"+working_directory_name)
    
    num_steps = 10 
    FOL_T_list = []
    T_c = coeffs_matrix[eval_id,:]
    FOL_T_list.append(T_c)
    for i in range(num_steps):
        FOL_T = np.array(fol.Predict(T_c.reshape(-1,1).T))
        FOL_T_list.append(FOL_T)
        T_c = FOL_T

    # solve FE here
    if solve_FE:
        first_fe_solver = FiniteElementSolver("first_fe_solver", thermal_loss_2d)
        start_time = time.process_time()
        T_c = Ts_c[eval_id]
        FE_T_list = []
        FE_T_list.append(T_c)
        for i in range(num_steps):
            FE_T = np.array(first_fe_solver.SingleSolve(T_c,np.zeros(fe_model.GetNumberOfNodes())))  
            FE_T_list.append(FE_T)
            T_c = FE_T
        print(f"\n############### FE solve took: {time.process_time() - start_time} s ###############\n")

        FOL_T_list = np.array(FOL_T_list)
        FE_T_list = np.array(FE_T_list)

        relative_error = abs(FOL_T.reshape(-1,1)- FE_T.reshape(-1,1))
        time_steps = [1,2,5,10]
        plot_mesh_vec_data_paper_temp([Ts_c[eval_id,:], FOL_T_list[time_steps[0]], FE_T_list[time_steps[0]]],['Heat Source', '$T$, FOL', '$T$, FEM'],f'sample_{time_steps[0]}')
        plot_mesh_vec_data_paper_temp([Ts_c[eval_id,:], FOL_T_list[time_steps[1]], FE_T_list[time_steps[1]]],['Heat Source', '$T$, FOL', '$T$, FEM'],f'sample_{time_steps[1]}')
        plot_mesh_vec_data_paper_temp([Ts_c[eval_id,:], FOL_T_list[time_steps[2]], FE_T_list[time_steps[2]]],['Heat Source', '$T$, FOL', '$T$, FEM'],f'sample_{time_steps[2]}')
        plot_mesh_vec_data_paper_temp([Ts_c[eval_id,:], FOL_T_list[time_steps[3]], FE_T_list[time_steps[3]]],['Heat Source', '$T$, FOL', '$T$, FEM'],f'sample_{time_steps[3]}')
        
        # eval_list = [eval_id2,eval_id3,eval_id4,eval_id5]#,eval_id6,eval_id7,eval_id8,eval_id9]
        # FOL_T = np.zeros((Ts_c[eval_id,:].reshape(-1,1).T).shape)
        # for i,eval_id in enumerate(eval_list):
        #     FOL_T = np.array(fol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T))
        #     # FOL_T = np.array(fol.Predict(coeffs_matrix[eval_id,:]))
        #     # print(f'eval coeffs: {coeffs_matrix[eval_id,:]}')
        #     # print(f"predicted array: {FOL_T}")
        #     start_time = time.process_time()
        #     FE_T = np.array(first_fe_solver.SingleSolve(Ts_c[eval_id],np.zeros(fe_model.GetNumberOfNodes())))  
        #     print(f"\n############### FE solve took: {time.process_time() - start_time} s ###############\n")
        #     plot_mesh_vec_data_paper_temp([Ts_c[eval_id,:], FOL_T, FE_T],['Heat Source', '$T$, FOL', '$T$, FEM'],f'sample_{eval_id}')
    
    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    fol_num_epochs = 1000
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
