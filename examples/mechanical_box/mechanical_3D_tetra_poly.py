import sys
import os

import numpy as np
from fol.computational_models.fe_model import FiniteElementModel
from fol.loss_functions.mechanical_3D_fe_tetra import MechanicalLoss3DTetra
from fol.solvers.fe_solver import FiniteElementSolver
from fol.IO.mesh_io import MeshIO
from fol.controls.voronoi_3D_control import VoronoiControl3D
from fol.deep_neural_networks.fe_operator_learning import FiniteElementOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle, time

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = "voronoi_box_3D_tetra"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))


    # import mesh
    point_bc_settings = {"Ux":{"left":0.0},
                        "Uy":{"left":0.0,"right":-0.05},
                        "Uz":{"left":0.0,"right":-0.05}}

    io = MeshIO("fol_io",'../meshes/',"box_3D_coarse.med",point_bc_settings)
    model_info = io.Import()

    # creation of fe model and loss function
    fe_model = FiniteElementModel("FE_model",model_info)
    mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_loss_3d",fe_model,{"young_modulus":1,"poisson_ratio":0.3})

    # voronoi control
    voronoi_control_settings = {"numberof_seeds":16,"k_rangeof_values":(0.1,1)}
    voronoi_control = VoronoiControl3D("voronoi_control",voronoi_control_settings,fe_model)

    # create some random coefficients & K for training
    create_random_coefficients = False
    if create_random_coefficients:
        number_of_samples = 200
        coeffs_matrix,K_matrix = create_random_voronoi_samples(voronoi_control,number_of_samples,dim=3)
        export_dict = {}
        export_dict["coeffs_matrix"] = coeffs_matrix
        with open(f'voronoi_3D_control_dict.pkl', 'wb') as f:
            pickle.dump(export_dict,f)
    else:
        with open(f'voronoi_3D_control_dict.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        
        coeffs_matrix = loaded_dict["coeffs_matrix"]

    K_matrix = voronoi_control.ComputeBatchControlledVariables(coeffs_matrix)

    # now save K matrix 
    export_Ks = False
    if export_Ks:
        for i in range(K_matrix.shape[0]):
            solution_file = os.path.join(case_dir, f"K_{i}.vtu")
            io.mesh_io.point_data['K'] = np.array(K_matrix[i,:])
            io.mesh_io.write(solution_file)

    # set NN hyper-parameters
    fol_num_epochs = 10
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

    first_fe_solver = FiniteElementSolver("first_fe_solver",mechanical_loss_3d)
    
    solution_file = os.path.join(case_dir, "voronoi_3D_box.vtu")

    # now we need to create, initialize and train fol
    fol = FiniteElementOperatorLearning("first_fol",voronoi_control,[mechanical_loss_3d],hidden_layer,
                                        "swish",load_NN_params=False,working_directory=working_directory_name)
    fol.Initialize()
    start_time = time.process_time()
    fol.Train(loss_functions_weights=[1],X_train=pc_train_mat,batch_size=fol_batch_size,num_epochs=fol_num_epochs,
                learning_rate=fol_learning_rate,optimizer="adam",convergence_criterion="total_loss",
                relative_error=1e-10,NN_params_save_file_name="NN_params_"+working_directory_name)
    print(f"\n############### fol train took: {time.process_time() - start_time} s ###############\n")

    if parametric_learning:
        UVW_train = fol.Predict(pc_train_mat)
        UVW_test = fol.Predict(pc_test_mat)

        test_eval_ids = [0]
        for eval_id in test_eval_ids:
            FOL_UVW = UVW_test[eval_id]
            io.mesh_io.point_data[f'UVW_test_FOL_{eval_id}'] = FOL_UVW.reshape((fe_model.GetNumberOfNodes(), 3))
            io.mesh_io.point_data[f'pc_test_pattern_{eval_id}'] = pc_test_nodal_value_matrix[eval_id].reshape((fe_model.GetNumberOfNodes(), 1))
            if solve_FE:
                start_time = time.process_time()
                FE_UVW = np.array(first_fe_solver.SingleSolve(pc_test_nodal_value_matrix[eval_id],np.zeros(3*fe_model.GetNumberOfNodes())))  
                print(f"\n############### FE solve took: {time.process_time() - start_time} s ###############\n")
                absolute_error = abs(FOL_UVW.reshape(-1,1)- FE_UVW.reshape(-1,1))
                
                io.mesh_io.point_data[f'UVW_test_FE_{eval_id}'] = FE_UVW.reshape((fe_model.GetNumberOfNodes(), 3))
                io.mesh_io.point_data[f'UVW_test_err_{eval_id}'] = absolute_error.reshape((fe_model.GetNumberOfNodes(), 3))

        train_eval_ids = [0]
        for eval_id in train_eval_ids:
            FOL_UV = UVW_train[eval_id]
            io.mesh_io.point_data[f'UVW_train_FOL_{eval_id}'] = FOL_UVW.reshape((fe_model.GetNumberOfNodes(), 3))
            io.mesh_io.point_data[f'pc_train_pattern_{eval_id}'] = pc_train_nodal_value_matrix[eval_id].reshape((fe_model.GetNumberOfNodes(), 1))
            if solve_FE:
                start_time = time.process_time()
                FE_UV = np.array(first_fe_solver.SingleSolve(pc_train_nodal_value_matrix[eval_id],np.zeros(3*fe_model.GetNumberOfNodes())))
                print(f"\n############### FE solve took: {time.process_time() - start_time} s ###############\n")
                absolute_error = abs(FOL_UVW.reshape(-1,1)- FE_UVW.reshape(-1,1))

                io.mesh_io.point_data[f'UVW_train_FE_{eval_id}'] = FE_UVW.reshape((fe_model.GetNumberOfNodes(), 3))
                io.mesh_io.point_data[f'UVW_train_err_{eval_id}'] = absolute_error.reshape((fe_model.GetNumberOfNodes(), 3))
            

    else:
        FOL_UVW = fol.Predict(pc_train_mat)
        io.mesh_io.point_data['UVW_train_FOL'] = FOL_UVW.reshape((fe_model.GetNumberOfNodes(), 3))
        io.mesh_io.point_data['pc_train_pattern'] = pc_train_nodal_value_matrix.reshape((fe_model.GetNumberOfNodes(), 1))
        if solve_FE:
            start_time = time.process_time()
            FE_UVW = np.array(first_fe_solver.SingleSolve(pc_train_nodal_value_matrix,np.zeros(3*fe_model.GetNumberOfNodes())))
            print(f"\n############### FE solve took: {time.process_time() - start_time} s ###############\n")
            absolute_error = abs(FOL_UVW.reshape(-1,1)- FE_UVW.reshape(-1,1))

            io.mesh_io.point_data['UVW_train_FE'] = FE_UVW.reshape((fe_model.GetNumberOfNodes(), 3))
            io.mesh_io.point_data['UVW_train_err'] = absolute_error.reshape((fe_model.GetNumberOfNodes(), 3))
        
    io.mesh_io.write(solution_file)

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
            print("Usage: python thermal_fol.py fol_num_epochs=10 solve_FE=False clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(fol_num_epochs, solve_FE,clean_dir)