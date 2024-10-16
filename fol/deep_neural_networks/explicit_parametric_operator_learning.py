"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/LICENSE
"""

from typing import Iterator, Tuple 
import jax
from tqdm import trange
import jax.numpy as jnp
from jax import random,jit,vmap
from functools import partial
import optax
from optax import GradientTransformation
from flax import nnx
from .deep_network import DeepNetwork
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *

class ExplicitParametricOperatorLearning(DeepNetwork):
    """
    A class for explicit parametric operator learning in deep neural networks.

    This class extends the `DeepNetwork` base class and is designed specifically 
    for learning parametric operators where spatial fields like predicted displacement
    are explicitly modeled. It inherits all the attributes and methods from `DeepNetwork` and introduces 
    additional components to handle control parameters.

    Attributes:
        name (str): The name assigned to the neural network model for identification purposes.
        control (Control): An instance of the Control class used for the parametric learning.
        loss_function (Loss): An instance of the Loss class representing the objective function to be minimized during training.
        flax_neural_network (Module): The Flax neural network model (inherited from flax.nnx.Module) that defines the architecture and forward pass of the network.
        optax_optimizer (GradientTransformation): The Optax optimizer used to compute and apply gradients during the training process.
        checkpoint_settings (dict): A dictionary of configurations used to manage checkpoints, saving model states and parameters during or after training. Defaults to an empty dictionary.
     
    """

    def __init__(self,
                 name:str,
                 control:Control,
                 loss_function:Loss,
                 flax_neural_network:nnx.Module,
                 optax_optimizer:GradientTransformation,
                 checkpoint_settings:dict={},
                 working_directory='.'
                 ):
        super().__init__(name,loss_function,flax_neural_network,
                         optax_optimizer,checkpoint_settings,
                         working_directory)
        self.control = control
        
    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize the explicit parametric operator learning model, its components, and control parameters.

        This method extends the initialization process defined in the `DeepNetwork` base class by
        ensuring that the control parameters used for parametric learning are also initialized.
        It handles both the initialization of core deep learning components (loss function, 
        checkpoint settings, neural network state restoration) and the initialization of 
        the control parameters essential for explicit parametric learning tasks.

        Parameters:
        ----------
        reinitialize : bool, optional
            If True, forces reinitialization of the model and its components, including control parameters,
            even if they have been initialized previously. Default is False.

        """

        if self.initialized and not reinitialize:
            return

        super().Initialize(reinitialize)

        if not self.control.initialized:
            self.control.Initialize(reinitialize)

        self.initialized = True

        # now check if the input output layers size match with 
        # loss and control sizes, this is explicit parametric learning
        if not hasattr(self.flax_neural_network, 'in_features'):
            fol_error(f"the provided flax neural netwrok does not have in_features "\
                      "which specifies the size of the input layer ") 

        if not hasattr(self.flax_neural_network, 'out_features'):
            fol_error(f"the provided flax neural netwrok does not have out_features "\
                      "which specifies the size of the output layer") 

        if self.flax_neural_network.in_features != self.control.GetNumberOfVariables():
            fol_error(f"the size of the input layer is {self.flax_neural_network.in_features} "\
                      f"does not match the size of control variables {self.control.GetNumberOfVariables()}")

        if self.flax_neural_network.out_features != self.loss_function.GetNumberOfUnknowns():
            fol_error(f"the size of the output layer is {self.flax_neural_network.out_features} " \
                      f" does not match the size of unknowns of the loss function {self.loss_function.GetNumberOfUnknowns()}")

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,x_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        nn_output = nn_model(x_set[0])
        control_output = self.control.ComputeControlledVariables(x_set[0])
        return self.loss_function.ComputeSingleLoss(control_output,nn_output)

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeBatchLossValue(self,batch_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        batch_losses,(batch_mins,batch_maxs,batch_avgs) = jax.vmap(self.ComputeSingleLossValue,(0,None))(batch_set,nn_model)
        loss_name = self.loss_function.GetName()
        total_mean_loss = jnp.mean(batch_losses)
        return total_mean_loss, ({loss_name+"_min":jnp.min(batch_mins),
                                         loss_name+"_max":jnp.max(batch_maxs),
                                         loss_name+"_avg":jnp.mean(batch_avgs),
                                         "total_loss":total_mean_loss})

    @partial(nnx.jit, static_argnums=(0,))
    def TrainStep(self, nn_model:nnx.Module, optimizer:nnx.Optimizer, batch_set:Tuple[jnp.ndarray, jnp.ndarray]):
        (batch_loss, batch_dict), batch_grads = nnx.value_and_grad(self.ComputeBatchLossValue,argnums=1,has_aux=True) \
                                                                    (batch_set,nn_model)
        optimizer.update(batch_grads)
        return batch_dict

    @print_with_timestamp_and_execution_time
    def Train(self, train_set:Tuple[jnp.ndarray, jnp.ndarray], test_set:Tuple[jnp.ndarray, jnp.ndarray] = (jnp.array([]), jnp.array([])), 
              batch_size:int=100, convergence_settings:dict={}, plot_settings:dict={}, save_settings:dict={}):

        self.default_convergence_settings = {"num_epochs":100,"convergence_criterion":"total_loss",
                                             "relative_error":1e-8,"absolute_error":1e-8}
        convergence_settings = UpdateDefaultDict(self.default_convergence_settings,convergence_settings)
        
        self.default_plot_settings = {"plot_list":["total_loss"],"plot_rate":1,"plot_save_rate":100}
        plot_settings = UpdateDefaultDict(self.default_plot_settings,plot_settings)

        self.default_save_settings = {"save_nn_model":True}
        save_settings = UpdateDefaultDict(self.default_save_settings,save_settings)

        def update_batch_history_dict(batches_hist_dict,batch_dict,batch_index):
            # fill the batch dict
            if batch_index == 0:
                for key, value in batch_dict.items():
                    batches_hist_dict[key] = [value]
            else:
                for key, value in batch_dict.items():
                    batches_hist_dict[key].append(value)

            return batches_hist_dict
        
        def update_history_dict(hist_dict,batch_hist_dict):
            for key, value in batch_hist_dict.items():
                if "max" in key:
                    batch_hist_dict[key] = [max(value)]
                elif "min" in key:
                    batch_hist_dict[key] = [min(value)]
                elif "avg" in key:
                    batch_hist_dict[key] = [sum(value)/len(value)]
                elif "total" in key:
                    batch_hist_dict[key] = [sum(value)]

            if len(hist_dict.keys())==0:
                hist_dict = batch_hist_dict
            else:
                for key, value in batch_hist_dict.items():
                    hist_dict[key].extend(value)

            return hist_dict

        train_history_dict = {}
        test_history_dict = {}
        pbar = trange(convergence_settings["num_epochs"])
        converged = False
        for epoch in pbar:
            train_set_hist_dict = {}
            test_set_hist_dict = {}
            # now loop over batches
            batch_index = 0 
            for batch_set in self.CreateBatches(train_set, batch_size):
                batch_dict = self.TrainStep(self.flax_neural_network,self.nnx_optimizer,batch_set)
                train_set_hist_dict = update_batch_history_dict(train_set_hist_dict,batch_dict,batch_index)

                if len(test_set[0])>0:
                    _,test_dict = self.ComputeBatchLossValue(test_set,self.flax_neural_network)
                    test_set_hist_dict = update_batch_history_dict(test_set_hist_dict,test_dict,batch_index)
                else:
                    test_dict = {}
                
                batch_index += 1

            train_history_dict = update_history_dict(train_history_dict,train_set_hist_dict)
            print_dict = {"train_loss":train_history_dict["total_loss"][-1]}
            if len(test_set[0])>0:
                test_history_dict = update_history_dict(test_history_dict,test_set_hist_dict)
                print_dict = {"train_loss":train_history_dict["total_loss"][-1],
                              "test_loss":test_history_dict["total_loss"][-1]}

            pbar.set_postfix(print_dict)

            # check converged
            converged = self.CheckConvergence(train_history_dict,convergence_settings)

            # plot the histories
            if (epoch>0 and epoch %plot_settings["plot_save_rate"] == 0) or converged:
                self.PlotHistoryDict(plot_settings,train_history_dict,test_history_dict)

            if epoch<convergence_settings["num_epochs"]-1 and converged:
                break    

    def CheckConvergence(self,train_history_dict:dict,convergence_settings:dict):
        convergence_criterion = convergence_settings["convergence_criterion"]
        absolute_error = convergence_settings["absolute_error"]
        relative_error = convergence_settings["relative_error"]
        num_epochs = convergence_settings["num_epochs"]
        current_epoch = len(train_history_dict[convergence_criterion])
        # check for absolute and relative errors and convergence
        if abs(train_history_dict[convergence_criterion][-1])<absolute_error:
            return True
        if current_epoch>1:
            if abs(train_history_dict[convergence_criterion][-1] -
                   train_history_dict[convergence_criterion][-2])<relative_error:
                return True
            elif current_epoch>=num_epochs:
                return True
            else:
                return False
        else:
            return False        

    def PlotHistoryDict(self,plot_settings:dict,train_history_dict:dict,test_history_dict:dict):
        
        plot_rate = plot_settings["plot_rate"]
        plot_list = plot_settings["plot_list"]
        plt.figure(figsize=(10, 5))
        for key,value in train_history_dict.items():
            if len(value)>0 and (len(plot_list)==0 or key in plot_list):
                plt.semilogy(value[::plot_rate], label=f"train_{key}") 

        for key,value in test_history_dict.items():
            if len(value)>0 and (len(plot_list)==0 or key in plot_list):
                plt.semilogy(value[::plot_rate], label=f"test_{key}") 
        plt.title("Training History")
        plt.xlabel(str(plot_rate) + " Epoch")
        plt.ylabel("Log Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.working_directory,"training_history.png"), bbox_inches='tight')
        plt.close()

    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,))
    def Predict(self,batch_X):
        def ForwardPassWithBC(x_input,NN_params):
            y_output = self.ForwardPass(x_input,NN_params)
            for loss_function in self.loss_functions:
                y_output_full = loss_function.GetFullDofVector(x_input,y_output)
            return y_output_full
        return jnp.squeeze(vmap(ForwardPassWithBC, (0,None))(batch_X,self.NN_params))

    def Finalize(self):
        pass