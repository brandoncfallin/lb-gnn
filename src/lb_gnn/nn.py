import numpy as np
from numpy.typing import NDArray

from .weights import proj


class NeuralNetwork:
    def __init__(
        self,
        nn_type: str,  # "dnn" or "gnn"
        num_outputs: int,
        num_weights: int,  # total weights at each agent
        weights_vec: NDArray,
        nn_schema: dict,
        gamma_matrix: NDArray,
        forgetting_factor: float,
        citw: float,
        theta_bar: float,
        dt: float,
    ):
        # params
        self.nn_type = nn_type
        self.nn_schema = nn_schema
        self.num_weights = num_weights
        self.dt = dt
        # gains
        self.gamma_matrix = gamma_matrix
        self.forgetting_factor = forgetting_factor
        self.citw = citw
        self.theta_bar = theta_bar
        # weight management
        self.weights = weights_vec
        self.neighborhood_weights = np.zeros(num_weights)
        # gradient management
        self.gradient_wrt_weights: NDArray
        self.gradient_correction: NDArray

    # update NN weights according to loss function, store in self.weights
    def update_neural_network_weights(self, loss: NDArray) -> None:
        """
        Updates self.weights in place using the developed adaptive update law and projection. This method is called externally, in the Agent object

        Args:
            loss (np.ndarray): loss as determined by the NN type (observer, controller, etc.). To be passed in from the Agent object.

        Returns:
            None: Updates weights in place
        """
        # adaptive update law
        weights_dot = self.gamma_matrix @ (
            self.gradient_correction.T @ loss
            - self.forgetting_factor * self.weights
            - np.array(self.citw) * self.neighborhood_weights
        )
        # parameter projection
        weights_dot = proj(
            theta_hat_dot=weights_dot,
            theta_hat=self.weights,
            gamma_matrix=self.gamma_matrix,
            theta_bar=self.theta_bar,
        )
        # euler update in place
        self.weights += self.dt * weights_dot
