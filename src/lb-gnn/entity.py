from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
import utils

if TYPE_CHECKING:
    from nn import NeuralNetwork

from numpy.typing import NDArray
from setup import Config


class Entity(ABC):
    def __init__(self, initial_pos, initial_vel, config):
        self.positions = np.zeros((3, config.time_steps))
        self.velocities = np.zeros((3, config.time_steps))
        self.accelerations = np.zeros((3, config.time_steps))
        self.positions[:, 0] = initial_pos
        self.velocities[:, 0] = initial_vel
        self.accelerations[:, 0] = np.zeros(3)
        self.dt = config.dt

    @abstractmethod
    def dynamics(self, step: int) -> np.ndarray: ...

    def update_plant(self, step: int) -> None:
        self.accelerations[:, step] = self.dynamics(step)
        self.velocities[:, step] = (
            self.velocities[:, step - 1] + self.dt * self.accelerations[:, step]
        )
        self.positions[:, step] = (
            self.positions[:, step - 1] + self.dt * self.velocities[:, step]
        )


class Agent(Entity):
    def __init__(
        self,
        config: Config,
        initial_pos: NDArray,
        initial_vel: NDArray,
        target: Entity,
        node_id: int,
        is_pinned: bool,
    ):
        # global params
        super().__init__(initial_pos, initial_vel, config)  # TO-DO: assign args here
        self.neighbors = {}
        self.config = config
        self.target = target
        self.node_id = node_id
        self.is_pinned = is_pinned
        # gains
        self.observer_gain = config.observer_gain
        self.controller_gain = config.controller_gain
        self.alpha_1 = config.alpha_1
        self.alpha_2 = config.alpha_2
        # self values
        self.target_estimate = np.zeros((3, config.time_steps))
        self.target_estimate_dot = np.zeros((3, config.time_steps))
        self.target_estimate_dot_dot = np.zeros((3, config.time_steps))
        self.target_estimate_error = np.zeros((3, config.time_steps))
        self.target_estimate_error_dot = np.zeros((3, config.time_steps))
        self.estimate_error = np.zeros((3, config.time_steps))
        self.estimate_error_dot = np.zeros((3, config.time_steps))
        self.position_tracking_error = np.zeros((3, config.time_steps))
        self.velocity_tracking_error = np.zeros((3, config.time_steps))
        self.control_input = np.zeros((3, config.time_steps))
        self.interaction_dynamics = np.zeros((3, config.time_steps))
        # neighborhood values
        self.neighborhood_estimate = np.zeros((3, config.time_steps))
        self.neighborhood_estimate_dot = np.zeros((3, config.time_steps))
        self.neighborhood_estimate_error = np.zeros((3, config.time_steps))
        self.neighborhood_estimate_error_dot = np.zeros((3, config.time_steps))
        self.neighborhood_target_estimate_error = np.zeros((3, config.time_steps))
        self.neighborhood_target_estimate_error_dot = np.zeros((3, config.time_steps))
        self.neighborhood_pos_error = np.zeros((3, config.time_steps))
        self.neighborhood_vel_error = np.zeros((3, config.time_steps))
        # nn objects
        self.nn_1: NeuralNetwork | None = None
        self.nn_2: NeuralNetwork | None = None
        # neural network inputs and outputs
        self.nn_1_input = np.zeros((config.nn_1_num_inputs, config.time_steps))
        self.nn_2_input = (
            np.zeros((2 * config.nn_2_num_inputs, config.time_steps))
            if config.nn_2_type == "dnn"
            else np.zeros((config.nn_2_num_inputs, config.time_steps))
        )
        self.nn_1_output = np.zeros((config.nn_1_num_outputs, config.time_steps))
        self.nn_2_output = np.zeros((config.nn_2_num_outputs, config.time_steps))
        # gradient management
        self.nn_1_gradient_correction: NDArray
        self.nn_2_gradient_correction: NDArray

    def measure(self, step: int) -> None:
        """
        Measures agent states using the latest values

        Args:
            step (int): Current time step

        Returns:
            None: Updates state values in place
        """

        self.position_tracking_error[:, step] = (
            self.target.positions[:, step] - self.positions[:, step - 1]
        )
        self.velocity_tracking_error[:, step] = (
            self.target.velocities[:, step] - self.velocities[:, step - 1]
        )
        self.estimate_error[:, step] = (
            self.target_estimate[:, step - 1] - self.positions[:, step - 1]
        )
        self.estimate_error_dot[:, step] = (
            self.target_estimate_dot[:, step - 1] - self.velocities[:, step - 1]
        )
        self.target_estimate_error[:, step] = (
            self.position_tracking_error[:, step] - self.estimate_error[:, step]
        )
        self.target_estimate_error_dot[:, step] = (
            self.velocity_tracking_error[:, step] - self.estimate_error_dot[:, step]
        )

    def neighborhood_measure(self, step: int) -> None:
        """
        Measures values from neighborhood agents for controller and observer updates

        Args:
            step (int): Current time step

        Returns:
            None: Updates neighborhood states in place
        """

        assert self.nn_1 is not None and self.nn_2 is not None
        # reset neighborhood_weights
        self.nn_1.neighborhood_weights = np.zeros_like(self.nn_1.neighborhood_weights)
        self.nn_2.neighborhood_weights = np.zeros_like(self.nn_2.neighborhood_weights)
        for neighbor in self.neighbors.values():
            # compute neighborhood sums
            self.neighborhood_pos_error[:, step] += (
                self.positions[:, step - 1] - neighbor.positions[:, step - 1]
            )
            self.neighborhood_vel_error[:, step] += (
                self.velocities[:, step - 1] - neighbor.velocities[:, step - 1]
            )
            self.neighborhood_estimate_error[:, step] += (
                neighbor.estimate_error[:, step] - self.estimate_error[:, step]
            )
            self.neighborhood_estimate_error_dot[:, step] += (
                neighbor.estimate_error_dot[:, step] - self.estimate_error_dot[:, step]
            )
            self.neighborhood_target_estimate_error[:, step] += (
                neighbor.target_estimate[:, step - 1]
                - self.target_estimate[:, step - 1]
            )
            self.neighborhood_target_estimate_error_dot[:, step] += (
                neighbor.target_estimate_dot[:, step - 1]
                - self.target_estimate_dot[:, step - 1]
            )
            self.neighborhood_estimate[:, step] += (
                neighbor.target_estimate[:, step - 1]
                - self.target_estimate[:, step - 1]
            )
            self.neighborhood_estimate_dot[:, step] += (
                neighbor.target_estimate_dot[:, step - 1]
                - self.target_estimate_dot[:, step - 1]
            )
            self.nn_1.neighborhood_weights += self.nn_1.weights - neighbor.nn_1.weights
            self.nn_2.neighborhood_weights += self.nn_2.weights - neighbor.nn_2.weights

    def integrate(self, step: int) -> None:
        """
        Integrates the observer state using provided acceleration values

        Args:
            step (int): Current time step

        Returns:
            None: Updates observer state in place
        """

        # integrate observer output
        self.target_estimate_dot[:, step] = (
            self.target_estimate_dot[:, step - 1]
            + self.dt * self.target_estimate_dot_dot[:, step]
        )
        self.target_estimate[:, step] = (
            self.target_estimate[:, step - 1]
            + self.dt * self.target_estimate_dot[:, step]
        )

    def observer(self, step: int) -> None:
        """
        Constructs observer state update

        Args:
            step (int): Current time step

        Returns:
            None: Updates observer state in place
        """

        self.target_estimate_dot_dot[:, step] = (
            self.observer_gain
            * (
                self.neighborhood_estimate_dot[:, step]
                + self.alpha_1 * self.neighborhood_estimate[:, step]
                + self.is_pinned
                * (
                    self.target_estimate_error_dot[:, step]
                    + self.alpha_1 * self.target_estimate_error[:, step]
                )
            )
            + self.nn_1_output[:, step]
            + self.nn_1_gradient_correction
        )

    def controller(self, step: int) -> None:
        """
        Updates controller state

        Args:
            step (int): Current time step

        Returns:
            None: Updates controller state in place
        """

        self.control_input[:, step] = (
            -self.nn_2_output[:, step]
            + self.target_estimate_dot_dot[:, step]
            + self.controller_gain
            * (
                self.estimate_error_dot[:, step]
                + self.alpha_2 * self.estimate_error[:, step]
            )
            - self.nn_2_gradient_correction
        )

    def update_law_1(self, step: int) -> None:
        """
        Performs weight update for nn_1 using the computed loss function

        Args:
            step (int): Current time step
        Returns:
            None: Updates nn_1 weights in place
        """

        # define loss function
        assert self.nn_1 is not None
        loss = (
            self.neighborhood_estimate_dot[:, step]
            + self.alpha_1 * self.neighborhood_estimate[:, step]
            + self.is_pinned
            * (
                self.target_estimate_error_dot[:, step]
                + self.alpha_1 * self.target_estimate_error[:, step]
            )
        )
        self.nn_1.update_neural_network_weights(loss=loss)

    def update_law_2(self, step: int) -> None:
        """
        Performs weight update for nn_2 using the computed loss function

        Args:
            step (int): Current time step
        Returns:
            None: Updates nn_2 weights in place
        """

        # define loss function
        assert self.nn_2 is not None
        loss = -(
            self.estimate_error_dot[:, step]
            + self.alpha_2 * self.estimate_error[:, step]
        )

        self.nn_2.update_neural_network_weights(loss=loss)

    def dynamics(self, step: int) -> np.ndarray:
        """
        Propagates Agent dynamics and returns (3,) array of x_ddot, y_ddot, z_ddot

        Args:
            step (int): Current time step
        Returns:
            np.ndarray: [x_ddot,y_ddot,z_ddot] values
        """

        self.controller(step)
        inter_agent = np.zeros(3)
        x = self.positions[0, step - 1]
        y = self.positions[1, step - 1]
        z = self.positions[2, step - 1]
        x_dot = self.velocities[0, step - 1]
        z_dot = self.velocities[2, step - 1]

        # neighborhood interaction
        for neighbor in self.neighbors.values():
            neigh_x = neighbor.positions[0, step - 1]
            neigh_y = neighbor.positions[1, step - 1]
            neigh_z = neighbor.positions[2, step - 1]
            neigh_x_dot = neighbor.velocities[0, step - 1]
            neigh_z_dot = neighbor.velocities[2, step - 1]
            inter_agent[0] += np.sin(y - neigh_y**2)  # + eps
            inter_agent[1] += 0.001 * (x - neigh_x) + (
                np.sin(((z_dot + z) * neigh_z_dot) * 0.02 * neigh_x_dot)
            )
            inter_agent[2] += (
                np.cos(z)
                * 0.05
                * (x_dot * neigh_x_dot)
                / np.sqrt(1 + np.abs(x_dot - neigh_x_dot))
            )

        self.interaction_dynamics[:, step] = inter_agent

        return self.control_input[:, step] + self.interaction_dynamics[:, step]


class Target(Entity):
    def __init__(self, initial_pos, initial_vel, config):
        super().__init__(initial_pos, initial_vel, config)

    def dynamics(self, step: int) -> np.ndarray:
        """
        Returns (3,) array of [x_ddot,y_ddot,z_ddot] for Target agent

        Args:
            step (int): Current time step

        Returns:
            np.ndarray: (3,) array of [x_ddot,y_ddot,z_ddot]
        """

        # coupled van der pol oscillator
        nonlinear_strength = 0.875
        coupling_strength = 0.1

        x, y, z = self.positions[:, step - 1]
        x_dot, y_dot, z_dot = self.velocities[:, step - 1]

        x_dot_dot = (
            nonlinear_strength * (1 - x**2) * x_dot - x + coupling_strength * (y - x)
        )
        y_dot_dot = (
            nonlinear_strength * (1 - y**2) * y_dot - y + coupling_strength * (z - y)
        )
        z_dot_dot = (
            nonlinear_strength * (1 - z**2) * z_dot - z + coupling_strength * (x - z)
        )

        return np.array([x_dot_dot, y_dot_dot, z_dot_dot])
