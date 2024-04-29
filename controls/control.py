from abc import ABC, abstractmethod
import jax.numpy as jnp

class Control(ABC):
    """Base abstract control class.

    The base abstract control class has the following responsibilities.
        1. Initalizes and finalizes the model.

    """
    def __init__(self, control_name: str) -> None:
        self.__name = control_name

    def GetName(self) -> str:
        return self.__name

    @abstractmethod
    def Initialize(self) -> None:
        """Initializes the control.

        This method initializes the control. This is only called once in the whole training process.

        """
        pass

    @abstractmethod
    def GetNumberOfVariables(self):
        """Returns number of variables of the control.

        """
        pass
    
    @abstractmethod
    def GetNumberOfControlledVariables(self):
        """Returns number of controlled variables

        """
        pass

    @abstractmethod
    def ComputeControlledVariables(self,variable_vector:jnp.array) -> None:
        """Computes the controlled variables for the given variables.

        """
        pass

    @abstractmethod
    def ComputeJacobian(self,variable_vector:jnp.array) -> None:
        """Computes jacobian of the control w.r.t input variable vector.

        """
        pass

    @abstractmethod
    def Finalize(self) -> None:
        """Finalizes the control.

        This method finalizes the control. This is only called once in the whole training process.

        """
        pass



