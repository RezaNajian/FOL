import numpy as np
from computational_models import FiniteElementModel
from loss_functions import MechanicalLoss
from solvers import FiniteElementSolver
from controls import FourierControl
from deep_neural_networks import FiniteElementOperatorLearning
from usefull_functions import *
# problem setup
L=1
N=21 
x_freqs = np.array([1,2,3])
y_freqs = np.array([1,2,3])
z_freqs = np.array([0])
model_info = create_2D_square_model_info_mechanical(L,N,Ux_left=0.0,Ux_right=0.1,Uy_left=0.0,Uy_right=0.1)

fe_model = FiniteElementModel("first_FE_model",model_info)
first_mechanical_loss = MechanicalLoss("first_mechanical_loss",fe_model)
first_fe_solver = FiniteElementSolver("first_fe_solver",first_mechanical_loss)
fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":2}
fourier_control = FourierControl("first_fourier_control",fourier_control_settings,fe_model)

# create some random coefficients & K for training
coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control)

# now we need to create, initialize and train fol
fol = FiniteElementOperatorLearning("first_fol",fourier_control,[first_mechanical_loss],[5,10],
                                    "swish",load_NN_params=False,NN_params_file_name="test.npy")
fol.Initialize()
fol.Train(loss_functions_weights=[1],X_train=coeffs_matrix,batch_size=20,num_epochs=1000,
          learning_rate=0.001,optimizer="adam",convergence_criterion="total_loss",
          relative_error=1e-7,save_NN_params=True,NN_params_save_file_name="test.npy")

fol.ReTrain(loss_functions_weights=[1],X_train=coeffs_matrix,batch_size=20,num_epochs=100,
            convergence_criterion="total_loss",relative_error=1e-7,save_NN_params=True,NN_params_save_file_name="test.npy")

FOL_UV_matrix = fol.Predict(coeffs_matrix)
FE_T_matrix = first_fe_solver.BatchSolve(K_matrix,np.zeros(FOL_UV_matrix.shape))

plot_mesh_vec_data(1,[K_matrix[100,:],FOL_UV_matrix[100,0::2],FOL_UV_matrix[1,1::2]],["K","U","V"],file_name="FOL-KUV-dist.png")
plot_mesh_vec_data(1,[K_matrix[100,:],FE_T_matrix[100,0::2],FE_T_matrix[1,1::2]],["K","U","V"],file_name="FE-KUV-dist.png")
