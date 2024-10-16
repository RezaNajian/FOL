"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/LICENSE
"""
import os
from abc import ABC,abstractmethod
from typing import Iterator,Tuple
from tqdm import trange
import matplotlib.pyplot as plt
import jax
from jax import jit
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from functools import partial
from flax import nnx
from optax import GradientTransformation
import orbax.checkpoint as ocp
from fol.loss_functions.loss import Loss
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *

class DeepNetwork(ABC):
    """
    Base abstract class for deep learning models.

    This class serves as a foundation for deep neural networks. It provides a 
    structure to initialize essential components such as the network, optimizer, 
    loss function, and checkpoint settings. The class is abstract and intended 
    to be extended by specific model implementations.

    Attributes:
    ----------
    name : str
        The name of the model, used for identification and checkpointing.
    loss_function : Loss
        The loss function that the model will optimize during training. 
        It defines the objective that the network is learning to minimize.
    flax_neural_network : nnx.Module
        The Flax neural network module that defines the model's architecture.
    optax_optimizer : GradientTransformation
        The Optax optimizer used to update model parameters during training.
    checkpoint_settings : dict, optional
        Dictionary that stores settings for saving and restoring checkpoints. 
        Defaults to an empty dictionary.

    """

    def __init__(self,
                 name:str,
                 loss_function:Loss,
                 flax_neural_network:nnx.Module,
                 optax_optimizer:GradientTransformation,
                 checkpoint_settings:dict={},
                 working_directory='.'):
        self.name = name
        self.loss_function = loss_function
        self.flax_neural_network = flax_neural_network
        self.optax_optimizer = optax_optimizer
        self.checkpoint_settings = checkpoint_settings
        self.working_directory = working_directory
        self.initialized = False
        self.default_checkpoint_settings = {"restore_state":False,
                                            "save_state":True,
                                            "state_directory":'./state'}

    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize the deep learning model, its components, and checkpoint settings.

        This method handles the initialization of essential components for the deep network. 
        It ensures that the loss function is initialized, sets up checkpointing 
        for saving and restoring model states, and manages reinitialization if needed. 
        The function is responsible for restoring the model's state from a previous checkpoint, 
        if specified in the checkpoint settings.

        Attributes:
        ----------
        reinitialize : bool, optional
            If True, forces reinitialization of the model and its components even if 
            they have been initialized previously. Default is False.

        Raises:
        -------
        AssertionError:
            If the restored neural network state does not match the current state 
            (based on a comparison using `np.testing.assert_array_equal`).
        """

        # initialize inputs
        if not self.loss_function.initialized:
            self.loss_function.Initialize(reinitialize)

        # create orbax checkpointer
        self.checkpointer = ocp.StandardCheckpointer()

        self.checkpoint_settings = UpdateDefaultDict(self.default_checkpoint_settings,
                                                     self.checkpoint_settings)
        
        # restore flax nn.Module from the file
        if self.checkpoint_settings["restore_state"]:

            state_directory = self.checkpoint_settings["state_directory"]
            absolute_path = os.path.abspath(state_directory)

            # get the state
            nn_state = nnx.state(self.flax_neural_network)

            # restore
            restored_state = self.checkpointer.restore(absolute_path, nn_state)

            # verify and cross check
            jax.tree.map(np.testing.assert_array_equal, nn_state, restored_state)

            # now update the model with the loaded state
            nnx.update(self.flax_neural_network, restored_state)

        # initialize the nnx optimizer
        self.nnx_optimizer = nnx.Optimizer(self.flax_neural_network, self.optax_optimizer)

    def GetName(self) -> str:
        return self.name
    
    def CreateBatches(self,data: Tuple[jnp.ndarray, jnp.ndarray], batch_size: int) -> Iterator[jnp.ndarray]:
        """Creates batches for the given inputs.

        """

        # Unpack data into data_x and data_y
        if len(data) > 1:
            data_x, data_y = data  
            if data_x.shape[0] != data_y.shape[0]:
                fol_error("data_x and data_y must have the same number of samples.")
        else:
            data_x = data[0]

        # Iterate over the dataset and yield batches of data_x and data_y
        for i in range(0, data_x.shape[0], batch_size):
            batch_x = data_x[i:i+batch_size, :]
            if len(data) > 1:
                batch_y = data_y[i:i+batch_size, :]
                yield batch_x, batch_y
            else:
                yield batch_x,

    @partial(jit, static_argnums=(0,))
    def StepAdam(self,opt_itr,opt_state,x_batch,NN_params):
        (total_loss, batch_dict), final_grads = self.ComputeTotalLossValueAndGrad(NN_params,x_batch)
        updated_state = self.opt_update(opt_itr, final_grads, opt_state)
        updated_NN_params = self.get_params(updated_state)
        return updated_NN_params,updated_state,batch_dict
    
    @partial(jit, static_argnums=(0,))
    def StepLBFGS(self,opt_itr,opt_state,x_batch,NN_params):
        updated_NN_params, updated_state = self.solver.update(params=NN_params, state=opt_state, batch_input=x_batch)
        return updated_NN_params,updated_state,updated_state.aux

    def Run(self,X_train,batch_size,num_epochs,convergence_criterion,relative_error,
            absolute_error,plot_list,plot_rate,plot_save_rate):
        
        pbar = trange(num_epochs)
        step_iteration = 0
        converged = False
        for epoch in pbar:
            batches_dict = {}
            # now loop over batches
            batch_index = 0 
            for batch in self.CreateBatches(X_train, batch_size):
                self.NN_params,self.opt_state,step_dict = self.step_function(step_iteration,self.opt_state,batch,self.NN_params)
                # fill the batch dict
                if batch_index == 0:
                    for key, value in step_dict.items():
                        batches_dict[key] = [value]
                else:
                    for key, value in step_dict.items():
                        batches_dict[key].append(value)
                step_iteration += 1
                batch_index += 1

            for key, value in batches_dict.items():
                if "max" in key:
                    batches_dict[key] = [max(value)]
                elif "min" in key:
                    batches_dict[key] = [min(value)]
                elif "avg" in key:
                    batches_dict[key] = [sum(value)/len(value)]
                elif "total" in key:
                    batches_dict[key] = [sum(value)]

            if not self.train_history_dict:
                self.train_history_dict.update(batches_dict)
            else:
                for key, value in batches_dict.items():
                    self.train_history_dict[key].extend(value)

            pbar.set_postfix({convergence_criterion: self.train_history_dict[convergence_criterion][-1]})

            # check for absolute and relative errors and convergence
            if self.train_history_dict[convergence_criterion][-1]<absolute_error:
                converged = True
            elif epoch>0 and abs(self.train_history_dict[convergence_criterion][-1] -
                 self.train_history_dict[convergence_criterion][-2])<relative_error:
                converged = True

            # plot the histories
            if epoch %plot_save_rate == 0 or epoch==num_epochs-1 or converged:
                self.PlotTrainHistory(plot_list,plot_rate)

            if converged:
                break

    def Train(self,X_train, batch_size=100, num_epochs=1000, learning_rate=0.01,optimizer="adam",
              convergence_criterion="total_loss",relative_error=1e-8,absolute_error=1e-8,plot_list=[],
              plot_rate=1,plot_save_rate=1000,save_NN_params=True, NN_params_save_file_name="NN_params.npy") -> None:
        """Trains the network.

        This method trains the network.

        """

        # here specify the optimizer
        if optimizer =="adam":
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
            self.opt_state = self.opt_init(self.NN_params)
            self.step_function = self.StepAdam
        elif optimizer=="LBFGS":
            self.solver = jaxopt.LBFGS(fun=self.ComputeTotalLossValueAndGrad,value_and_grad=True,has_aux=True,stepsize=-1,
                                       linesearch="backtracking",stop_if_linesearch_fails=True,maxiter=num_epochs,verbose=False)
            self.opt_state = self.solver.init_state(init_params=self.NN_params,batch_input=X_train)
            self.step_function = self.StepLBFGS

        # now run the training
        self.Run(X_train,batch_size,num_epochs,convergence_criterion,
                 relative_error,absolute_error,plot_list,plot_rate,plot_save_rate)

        # save optimized NN parameters
        if save_NN_params:
            flat_params, _ = ravel_pytree(self.NN_params)
            jnp.save(os.path.join(self.working_directory,NN_params_save_file_name), flat_params)

    def ReTrain(self,X_train,batch_size=100,num_epochs=1000,convergence_criterion="total_loss",
                relative_error=1e-8,absolute_error=1e-8,reset_train_history=False,plot_list=[],
                plot_rate=1,plot_save_rate=1000,save_NN_params=True,NN_params_save_file_name="NN_params.npy") -> None:
        """ReTrains the network.

        This method retrains the network.

        """
        if reset_train_history:
            self.train_history_dict = {}

        # now run the training
        self.Run(X_train,batch_size,num_epochs,convergence_criterion,relative_error,
                 absolute_error,plot_list,plot_rate,plot_save_rate)

        # save optimized NN parameters
        if save_NN_params:
            flat_params, _ = ravel_pytree(self.NN_params)
            jnp.save(os.path.join(self.working_directory,NN_params_save_file_name), flat_params)

    @abstractmethod
    def Finalize(self) -> None:
        """Finalizes the network.

        This method finalizes the network. This is only called once in the whole training process.

        """
        pass





