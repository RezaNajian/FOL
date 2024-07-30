"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/License.txt
"""
from .deep_network import DeepNetwork 
import jax
import jax.numpy as jnp
from jax.nn import relu,sigmoid,swish,tanh,leaky_relu,elu
from jax import random,jit,vmap
from functools import partial
from fol.tools.decoration_functions import *

class ImplicitOperatorLearning(DeepNetwork):
    @print_with_timestamp_and_execution_time
    def __init__(self,NN_name:str,loss_functions:list,hidden_layers:list,conductivity:jnp.array,omega:float,
                 load_NN_params:bool=False,NN_params_file_name:str=None,working_directory='.'):
        super().__init__(NN_name,load_NN_params,NN_params_file_name,working_directory)
        self.input_size = 3
        self.output_size = 1
        self.conductivity = conductivity
        self.hidden_layers = hidden_layers
        self.loss_functions = loss_functions
        self.omega = omega

    @print_with_timestamp_and_execution_time
    def Initialize(self):
        self.InitializeParameters()
        self.total_number_of_NN_params = self.flatten_NN_data(self.NN_params).shape[-1]
        for loss_func in self.loss_functions:
            loss_func.Initialize() 
        self.initialized = True

    def InitializeParameters(self):

        siren_layers = [self.hidden_layers[0]] + self.hidden_layers

        layer_sizes = [self.input_size] +  siren_layers + [self.output_size]
        key = random.PRNGKey(0)
        keys = random.split(key, len(layer_sizes) - 1)
        self.NN_params = []

        for i, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            weight_key, bias_key = random.split(keys[i])
            if i==0:
                weight_variance = 1 / in_dim
            else:
                weight_variance = jnp.sqrt(6 / in_dim) / self.omega
            
            weights = random.uniform(weight_key, (in_dim, out_dim), jnp.float32, minval=-weight_variance, maxval=weight_variance)
            bias_variance = jnp.sqrt(1 / in_dim)
            biases = random.uniform(bias_key, (int(out_dim),), jnp.float32, minval=-bias_variance, maxval=bias_variance)
            self.NN_params.append((weights, biases))

        super().InitializeParameters()
        
    @partial(jit, static_argnums=(0,))
    def ForwardPass(self,x,NN_params):
        for (w, b) in NN_params[:-1]:
            x = jnp.dot(x, w) + b
            x = jnp.sin(self.omega*x)
        final_w, final_b = NN_params[-1]
        x = jnp.dot(x, final_w) + final_b
        return x
    
    @partial(jit, static_argnums=(0,))
    def flatten_NN_data(self,NN_data):
        flat_params = []
        for w_d, b_d in NN_data:
            flat_params.append(w_d.flatten())
            flat_params.append(b_d.flatten())
        return jnp.concatenate(flat_params)

    @partial(jit, static_argnums=(0,))
    def unflatten_data_into_NN(self,flat_data):
        # Reconstruct the data from the flattened vector
        unflat_data = []
        idx = 0
        for weights, biases in self.NN_params:
            n_in, n_out = weights.shape
            num_w = n_in * n_out
            num_b = n_out

            new_weights = flat_data[idx:idx + num_w].reshape(n_in, n_out)
            idx += num_w

            new_biases = flat_data[idx:idx + num_b]
            idx += num_b

            unflat_data.append((new_weights, new_biases))
        return unflat_data

    @partial(jit, static_argnums=(0,))
    def ComputeTotalLossValueAndGrad(self,NN_params,batch_input):
        total_loss = 0.0
        total_loss_grads = jnp.zeros((self.total_number_of_NN_params))
        losses_dict = {}

        def ComputeLossValue(batch_input,NN_params,loss_index):
            batch_output = vmap(self.ForwardPass, (0,None))(batch_input,NN_params)
            K_temp_vals = jnp.ones(batch_output.shape)
            loss_name = self.loss_functions[loss_index].GetName()
            loss_val,(min_val,max_val,avg_val) = self.loss_functions[loss_index].ComputeSingleLoss(self.conductivity.reshape(-1),batch_output.reshape(-1))
            return loss_val, ({loss_name+"_min":min_val,loss_name+"_max":max_val,loss_name+"_avg":avg_val})        

        for loss_index,loss_func in enumerate(self.loss_functions):
            (batch_loss,(batch_dict)),batch_loss_grads = jax.value_and_grad(ComputeLossValue,argnums=1,has_aux=True)(batch_input,NN_params,loss_index)
            losses_dict.update(batch_dict)
            loss_weight = self.loss_functions_weights[loss_index]
            total_loss += loss_weight * batch_loss
            flat_loss_grads = self.flatten_NN_data(batch_loss_grads)
            flat_loss_grads /= jnp.linalg.norm(flat_loss_grads,ord=2)
            total_loss_grads = jnp.add(total_loss_grads,loss_weight * flat_loss_grads)
        
        total_loss_grads /= jnp.linalg.norm(total_loss_grads,ord=2)
        final_grads = self.unflatten_data_into_NN(total_loss_grads)

        return (total_loss, {**losses_dict,"total_loss":total_loss}), final_grads

    @print_with_timestamp_and_execution_time
    def Train(self, loss_functions_weights:list, X_train, batch_size=100, num_epochs=1000, learning_rate=0.01, 
              optimizer="adam",convergence_criterion="true_loss",relative_error=1e-6,absolute_error=1e-6, plot_list=[],
              plot_rate=1,plot_save_rate=1000,save_NN_params=True, NN_params_save_file_name="Fourier_FOL_Thermal_params.npy"):

        if len(loss_functions_weights) != len(self.loss_functions):
            raise ValueError(f"Number of loss functions weights do not match with number of loss functions !")

        self.loss_functions_weights = loss_functions_weights
        super().Train(X_train,batch_size,num_epochs,learning_rate, 
              optimizer,convergence_criterion,relative_error,absolute_error,plot_list,
              plot_rate,plot_save_rate,save_NN_params, NN_params_save_file_name)
    
    @print_with_timestamp_and_execution_time    
    def ReTrain(self, loss_functions_weights:list, X_train, batch_size=100, num_epochs=1000,  
                convergence_criterion="true_loss",relative_error=1e-6,absolute_error=1e-6,reset_train_history=False,
                plot_list=[],plot_rate=1,plot_save_rate=1000,save_NN_params=True, NN_params_save_file_name="Fourier_FOL_Thermal_params.npy"):

        if len(loss_functions_weights) != len(self.loss_functions):
            raise ValueError(f"Number of loss functions weights do not match with number of loss functions !")

        self.loss_functions_weights = loss_functions_weights
        super().ReTrain(X_train,batch_size,num_epochs,convergence_criterion,
                        relative_error,absolute_error,reset_train_history,
                        plot_list,plot_rate,plot_save_rate,save_NN_params, 
                        NN_params_save_file_name)

    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,))
    def Predict(self,batch_X):
        prediction = jnp.squeeze(vmap(self.ForwardPass, (0,None))(batch_X,self.NN_params))
        for loss_function in self.loss_functions:
            prediction = loss_function.ApplyBCOnDOFs(prediction)
        return prediction

    def Finalize(self):
        pass