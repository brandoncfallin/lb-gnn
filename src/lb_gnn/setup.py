from typing import Optional

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from .utils import construct_switching_steps


class Config:
    def __init__(
        self,
        # global settings
        trial_number: int,
        nn_1_type: str,  # DNN, GNN
        nn_2_type: str,  # DNN. GNN
        dt: float,
        tf: float,
        num_agents: int,
        topology: str,  # ring, star, path, complete, acyclic, switching
        pinned_agents: list[int],  # stay constant
        # switching
        switching_rng: Generator,
        mode_lower_bound: float,
        mode_upper_bound: float,
        block_upper_bound: float,
        switching_mode_order: list[int],
        switching_edge_sets: dict[int, list[tuple[int, int]]],
        # gains
        observer_gain: float,
        controller_gain: float,
        nn_1_forgetting_factor: float,
        nn_2_forgetting_factor: float,
        nn_1_citw: float,
        nn_2_citw: float,
        alpha_1: float,
        alpha_2: float,
        # target agent settings
        target_init_pos: NDArray,
        target_init_vel: NDArray,
        # global neural network settings
        num_neurons_per_layer: int,
        num_mlp_layers: int,
        num_message_passing_layers: int,
        same_weights_at_each_agent: bool,
        # neural network 1 settings (observer)
        nn_1_num_inputs: int,
        nn_1_num_outputs: int,
        nn_1_theta_bar: float,
        nn_1_gamma_params: dict,
        # neural network 2 settings (controller)
        nn_2_num_inputs: int,
        nn_2_num_outputs: int,
        nn_2_theta_bar: float,
        nn_2_gamma_params: dict,
        # rng
        weight_seed: Optional[int],
    ):
        # global settings
        self.trial_number = trial_number
        self.nn_1_type = nn_1_type
        self.nn_2_type = nn_2_type
        self.dt = dt
        self.tf = tf
        self.time_steps = int(tf / dt)
        self.num_agents = num_agents
        self.topology = topology
        self.pinned_agents = pinned_agents
        # switching
        self.switching_rng = switching_rng
        self.mode_lower_bound = mode_lower_bound
        self.mode_upper_bound = mode_upper_bound
        self.block_upper_bound = block_upper_bound
        self.switching_mode_order = switching_mode_order
        self.switching_edge_sets = switching_edge_sets
        self.switching_steps = construct_switching_steps(
            dt=dt,
            tf=tf,
            mode_lower_bound=mode_lower_bound,
            mode_upper_bound=mode_upper_bound,
            block_upper_bound=block_upper_bound,
            switching_mode_order=switching_mode_order,
            rng=switching_rng,
        )
        # gains
        self.observer_gain = observer_gain
        self.controller_gain = controller_gain
        self.nn_1_forgetting_factor = nn_1_forgetting_factor
        self.nn_2_forgetting_factor = nn_2_forgetting_factor
        self.nn_1_citw = nn_1_citw
        self.nn_2_citw = nn_2_citw
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        # target agent settings
        self.target_init_pos = target_init_pos
        self.target_init_vel = target_init_vel
        # global neural network settings
        self.num_neurons_per_layer = num_neurons_per_layer
        self.num_mlp_layers = num_mlp_layers
        self.num_message_passing_layers = num_message_passing_layers
        self.same_weights_at_each_agent = same_weights_at_each_agent
        # neural network 1 settings (for observer)
        self.nn_1_num_inputs = nn_1_num_inputs
        self.nn_1_num_outputs = nn_1_num_outputs
        self.nn_1_theta_bar = nn_1_theta_bar
        self.nn_1_gamma_params = nn_1_gamma_params
        self.nn_1_schema: dict
        self.nn_1_num_weights: int
        # neural network 2 settings (for controller)
        self.nn_2_num_inputs = nn_2_num_inputs
        self.nn_2_num_outputs = nn_2_num_outputs
        self.nn_2_theta_bar = nn_2_theta_bar
        self.nn_2_gamma_params = nn_2_gamma_params
        self.nn_2_schema: dict
        self.nn_2_num_weights: int
        # rng
        self.weight_seed = weight_seed
        self.rng = np.random.default_rng(seed=weight_seed)

    def initialize_nn_schemas(self) -> None:
        """
        Creates NN schemas for nn_1 and nn_2

        Args:
            None

        Returns:
            None: Updates values in place
        """

        self.nn_1_schema = self.create_nn_schema(
            nn_type=self.nn_1_type,
            is_graph_valued=False,
            num_message_passing_layers=self.num_message_passing_layers,
            num_mlp_layers=self.num_mlp_layers,
            num_neurons_per_layer=self.num_neurons_per_layer,
            num_inputs=self.nn_1_num_inputs,
            num_outputs=self.nn_1_num_outputs,
            num_agents=self.num_agents,
        )
        self.nn_1_num_weights = self.compute_num_weights(nn_schema=self.nn_1_schema)
        # nn_2
        self.nn_2_schema = self.create_nn_schema(
            nn_type=self.nn_2_type,
            is_graph_valued=True if self.nn_2_type == "dnn" else False,
            num_message_passing_layers=self.num_message_passing_layers,
            num_mlp_layers=self.num_mlp_layers,
            num_neurons_per_layer=self.num_neurons_per_layer,
            num_inputs=self.nn_2_num_inputs,
            num_outputs=self.nn_2_num_outputs,
            num_agents=self.num_agents,
        )
        self.nn_2_num_weights = self.compute_num_weights(nn_schema=self.nn_2_schema)

    def build_mlp_layers(
        self,
        num_mlp_layers: int,
        num_neurons_per_layer: int,
        num_inputs: int,
        num_outputs: int,
    ) -> list[tuple[int, int]]:
        """
        Assembles multi-layer perceptron (MLP) layer schema

        Args:
            num_mlp_layers (int): Total number of MLP layers, incl. output layer. Must be >=2
            num_neurons_per_layer (int): Number of neurons at each hidden layer
            num_inputs (int): Number of inputs to MLP
            num_outputs (int): Number of outputs of MLP

        Returns:
            list[tuple[int,int]]: List of (input_dim,output_dim) tuples for each MLP layer
        """

        assert num_mlp_layers >= 2, (
            "num_mlp_layers must be at least 2 (one hidden + one output)"
        )
        size_tuple_list = []
        for mlp_layer_idx in range(num_mlp_layers):
            if mlp_layer_idx == 0:  # input layer
                size_tuple_list.append((num_neurons_per_layer, num_inputs + 1))
            elif mlp_layer_idx != 0 and mlp_layer_idx != num_mlp_layers - 1:
                size_tuple_list.append(
                    (num_neurons_per_layer, num_neurons_per_layer + 1)
                )
            else:
                size_tuple_list.append((num_outputs, num_neurons_per_layer + 1))
        return size_tuple_list

    def create_nn_schema(
        self,
        nn_type: str,  # "dnn" or "gnn"
        is_graph_valued: bool,
        num_message_passing_layers: int,
        num_mlp_layers: int,
        num_neurons_per_layer: int,
        num_inputs: int,
        num_outputs: int,
        num_agents: int,
    ) -> dict:
        """
        Creates schema (pytree) for the DNN or GNN, to eventually be used to store weights and Jacobian values

        Args:
            nn_type (str): Indicates NN type, either "dnn" or "gnn"
            is_graph_valued (bool): Indicates if input values are stacked for whole graph
            num_message_passing_layers (int): Number of GNN message passing layers, always 1 for DNN
            num_mlp_layers (int): Total number of MLP layers, incl. output layer. Must be >=2
            num_neurons_per_layer (int): Number of neurons at each hidden layer
            num_inputs (int): Number of inputs to MLP
            num_outputs (int): Number of outputs of MLP
            num_agents (int): Number of agents in MAS
        """

        assert nn_type == "dnn" or nn_type == "gnn", "nn_type must be 'dnn' or 'gnn'"
        nn_schema = {}
        if nn_type == "dnn":
            for message_passing_layer_idx in range(1):
                nn_schema[message_passing_layer_idx] = {
                    "mlp_inner": self.build_mlp_layers(
                        num_mlp_layers=num_mlp_layers,
                        num_neurons_per_layer=num_neurons_per_layer,
                        num_inputs=num_inputs
                        if not is_graph_valued
                        else num_inputs * num_agents,
                        num_outputs=num_outputs,
                    ),
                    "mlp_outer": [],
                }
        else:
            for message_passing_layer_idx in range(num_message_passing_layers):
                if (
                    message_passing_layer_idx < num_message_passing_layers - 1
                    and num_message_passing_layers > 1
                ):
                    nn_schema[message_passing_layer_idx] = {
                        "mlp_inner": self.build_mlp_layers(
                            num_mlp_layers=num_mlp_layers,
                            num_neurons_per_layer=num_neurons_per_layer,
                            num_inputs=num_inputs,
                            num_outputs=num_inputs,
                        ),
                        "mlp_outer": self.build_mlp_layers(
                            num_mlp_layers=num_mlp_layers,
                            num_neurons_per_layer=num_neurons_per_layer,
                            num_inputs=2 * num_inputs,
                            num_outputs=num_inputs,
                        ),
                    }
                elif (
                    message_passing_layer_idx == num_message_passing_layers - 1
                    and num_message_passing_layers > 1
                ):
                    nn_schema[message_passing_layer_idx] = {
                        "mlp_inner": self.build_mlp_layers(
                            num_mlp_layers=num_mlp_layers,
                            num_neurons_per_layer=num_neurons_per_layer,
                            num_inputs=num_inputs,
                            num_outputs=num_inputs,
                        ),
                        "mlp_outer": self.build_mlp_layers(
                            num_mlp_layers=num_mlp_layers,
                            num_neurons_per_layer=num_neurons_per_layer,
                            num_inputs=2 * num_inputs,
                            num_outputs=num_outputs,
                        ),
                    }
                else:
                    nn_schema[message_passing_layer_idx] = {
                        "mlp_inner": self.build_mlp_layers(
                            num_mlp_layers=num_mlp_layers,
                            num_neurons_per_layer=num_neurons_per_layer,
                            num_inputs=num_inputs,
                            num_outputs=num_inputs,
                        ),
                        "mlp_outer": self.build_mlp_layers(
                            num_mlp_layers=num_mlp_layers,
                            num_neurons_per_layer=num_neurons_per_layer,
                            num_inputs=2 * num_inputs,
                            num_outputs=num_outputs,
                        ),
                    }

        return nn_schema

    def compute_num_weights(self, nn_schema: dict) -> int:
        """
        Computes total number of parameters for DNN or GNN

        Args:
            nn_schema (dict): Schema of appropriate form containing size tuples

        Returns:
            int: Total number of tunable parameters
        """

        num_weights = 0
        for message_passing_layer_idx in nn_schema.keys():
            # mlp_inner always exists
            for size_tuple in nn_schema[message_passing_layer_idx]["mlp_inner"]:
                num_weights += size_tuple[0] * size_tuple[1]
            # check if mlp_outer
            if nn_schema[message_passing_layer_idx]["mlp_outer"] != []:
                for size_tuple in nn_schema[message_passing_layer_idx]["mlp_outer"]:
                    num_weights += size_tuple[0] * size_tuple[1]
        return num_weights

    def compute_output_layer_weights(self, nn_type: str, nn_schema: dict) -> int:
        """
        Computes number of weights in output layer, used when plotting this specific set of weights

        Args:
            nn_type (str): "dnn" or "gnn"
            nn_schema (dict): Schema of appropriate form containing size tuples

        Returns:
            int: Number of tunable parameters in output layer
        """

        if nn_type == "dnn":
            # get last entry from first mpl
            size_tuple = nn_schema[0]["mlp_inner"][-1]
            return size_tuple[0] * size_tuple[1]
        else:
            # get last entry from mlp outer in last mpl
            message_passing_layer_indices = sorted(nn_schema.keys())
            last_index = message_passing_layer_indices[-1]
            size_tuple = nn_schema[last_index]["mlp_outer"][-1]
            return size_tuple[0] * size_tuple[1]
