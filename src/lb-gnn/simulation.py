import copy
import time

import jax
import jax.numpy as jnp
import numpy as np

import utils
from entity import Agent, Target
from nn import NeuralNetwork
from passes import build_jacobian_map, dnn_forward, gnn_forward
from setup import Config
from weights import (
    build_gamma_matrix,
    flatten_all_weights,
    initialize_weights,
    unflatten_all_weights,
)
from typing import Callable

class Simulation:
    def __init__(self, config: Config):
        self.config = config
        self.trial_number = config.trial_number  # for repeat trials
        # init schemas
        self.pinning_vector = utils.make_pin(
            num_agents=self.config.num_agents, pinned_agents=self.config.pinned_agents
        )
        self.edge_set = utils.make_edge_set(
            num_agents=self.config.num_agents, topology=self.config.topology
        )
        self.adjacency_matrix, self.augmented_adjacency_matrix = (
            utils.make_adjacency_matrix(
                num_agents=self.config.num_agents, edge_set=self.edge_set
            )
        )
        self.agents, self.target = self.initialize_agents()
        # nn data storage
        (
            self.nn_1_weights_dict,
            self.nn_2_weights_dict,
            self.nn_1_weights_array,
            self.nn_2_weights_array,
            self.nn_1_gamma_matrix,
            self.nn_2_gamma_matrix,
            self.nn_1_jacobian_map,
            self.nn_2_jacobian_map,
        ) = self.initialize_nns()
        self.assign_agent_nns()
        # error and result variables
        self.nn_1_agent_output_weights_trajectory = np.zeros(
            (self.config.nn_1_num_weights, self.config.time_steps)
        )
        self.nn_2_agent_output_weights_trajectory = np.zeros(
            (self.config.nn_2_num_weights, self.config.time_steps)
        )
        self.label = ""

    def initialize_agents(self) -> tuple[list[Agent],Target]:
        """
        Arranges agents in an N-gon od radius 10 and creates all Agent objects

        Args: None

        Returns:
            tuple[list[Agent],Target]: Tuple containing a list of Agents in ascending order of index and single Target object
        """
        # arrange agents in N-gon, radius 10
        radius = 3
        angles = np.linspace(0, 2 * np.pi, self.config.num_agents, endpoint=False)
        positions = np.column_stack(
            [
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.zeros(self.config.num_agents),
            ]
        )
        # create agents and target
        target = Target(
            initial_pos=self.config.target_init_pos,
            initial_vel=self.config.target_init_vel,
            config=self.config,
        )

        agents = [
            Agent(
                initial_pos=positions[i],
                initial_vel=np.zeros(3),
                config=self.config,
                target=target,
                node_id=i,
                is_pinned=bool(self.pinning_vector[i]),  # link agent to target if true
            )
            for i in range(self.config.num_agents)
        ]
        # construct neighbors dict for each agent
        for i, j in self.edge_set:
            if i != j:  # Only add if not self
                agents[j].neighbors[i] = agents[i]  # add to dict
        return agents, target

    def initialize_nns(self) -> tuple[dict,dict,np.ndarray,np.ndarray,np.ndarray,np.ndarray,Callable,Callable]:
        """
        Generates weights and adaptation gains for the observer and controller NNs

        Args: None

        Returns:
            tuple[dict,dict,np.ndarray,np.ndarray,np.ndarray,np.ndarray,Callable,Callable]: Tuple containing weights dictionaries, weights arrays, adapation gain matrices, and Jax Jacobian maps for nn_1 and nn_2
        """
        # initialize NN weights
        nn_1_weights_dict = initialize_weights(
            num_agents=self.config.num_agents,
            nn_schema=self.config.nn_1_schema,
            rng=self.config.rng,
            same_at_each_agent=self.config.same_weights_at_each_agent,
        )
        nn_2_weights_dict = initialize_weights(
            num_agents=self.config.num_agents,
            nn_schema=self.config.nn_2_schema,
            rng=self.config.rng,
            same_at_each_agent=self.config.same_weights_at_each_agent,
        )

        # initialize NN gamma matrices
        nn_1_gamma_matrix = build_gamma_matrix(
            gamma_params=self.config.nn_1_gamma_params,
            nn_schema=self.config.nn_1_schema,
        )
        nn_2_gamma_matrix = build_gamma_matrix(
            gamma_params=self.config.nn_2_gamma_params,
            nn_schema=self.config.nn_2_schema,
        )

        # initialize NN jacobian maps
        nn_1_jacobian_map = build_jacobian_map(
            forward_pass=dnn_forward
            if self.config.nn_1_type == "dnn"
            else gnn_forward,  
            adjacency_matrix=jnp.array(self.adjacency_matrix),
            augmented_adjacency_matrix=jnp.array(self.augmented_adjacency_matrix),
            num_agents=self.config.num_agents,
            input_dim=self.config.nn_1_num_inputs,
            is_graph_valued=False,  
        )
        nn_2_jacobian_map = build_jacobian_map(
            forward_pass=dnn_forward
            if self.config.nn_2_type == "dnn"
            else gnn_forward,  
            adjacency_matrix=jnp.array(self.adjacency_matrix),
            augmented_adjacency_matrix=jnp.array(self.augmented_adjacency_matrix),
            num_agents=self.config.num_agents,
            input_dim=self.config.nn_2_num_inputs,
            is_graph_valued=True,  
        )

        # flatten weights for assignment
        nn_1_weights_array = flatten_all_weights(
            num_agents=self.config.num_agents,
            num_weights=self.config.nn_1_num_weights,
            nn_weights_dict=nn_1_weights_dict,
        )
        nn_2_weights_array = flatten_all_weights(
            num_agents=self.config.num_agents,
            num_weights=self.config.nn_2_num_weights,
            nn_weights_dict=nn_2_weights_dict,
        )

        return (
            nn_1_weights_dict,
            nn_2_weights_dict,
            nn_1_weights_array,
            nn_2_weights_array,
            nn_1_gamma_matrix,
            nn_2_gamma_matrix,
            nn_1_jacobian_map,
            nn_2_jacobian_map,
        )

    def assign_agent_nns(self) -> None:
        """
        Gives NN objects to each agent for controller and observer NNs

        Args: None

        Returns: None
        """
        # initialize NNs
        for agent_idx in range(self.config.num_agents):
            self.agents[agent_idx].nn_1 = NeuralNetwork(
                nn_type=self.config.nn_1_type,
                num_outputs=self.config.nn_1_num_outputs,
                num_weights=self.config.nn_1_num_weights,
                weights_vec=self.nn_1_weights_array[agent_idx, :],
                nn_schema=self.config.nn_1_schema,
                gamma_matrix=self.nn_1_gamma_matrix,
                forgetting_factor=self.config.nn_1_forgetting_factor,
                citw=self.config.nn_1_citw,
                theta_bar=self.config.nn_1_theta_bar,
                dt=self.config.dt,
            )
            self.agents[agent_idx].nn_2 = NeuralNetwork(
                nn_type=self.config.nn_2_type,
                num_outputs=self.config.nn_2_num_outputs,
                num_weights=self.config.nn_2_num_weights,
                weights_vec=self.nn_2_weights_array[agent_idx, :],
                nn_schema=self.config.nn_2_schema,
                gamma_matrix=self.nn_2_gamma_matrix,
                forgetting_factor=self.config.nn_2_forgetting_factor,
                citw=self.config.nn_2_citw,
                theta_bar=self.config.nn_2_theta_bar,
                dt=self.config.dt,
            )

    def run(self) -> None:
        """
        Executes simulation and prints progress every 100 time steps

        Args: None

        Returns: None
        """
        start = time.time()

        print(f"num_weights: {self.config.nn_1_num_weights}")

        for i in range(1, self.config.time_steps):
            self.update_sim(step=i)
            # every 100 time steps print status
            if i % 100 == 0:
                print(
                    f"Progress: {i / self.config.time_steps * 100:.2f}%",
                    end="\r",
                    flush=True,
                )
        end = time.time()
        print(self.label + f" simulation complete. Time elapsed: {end - start:.2f}s")

    def update_sim(self, step: int) -> None:
        """
        Propagates the simulation forward one step in time

        Note:
            Current flow is measure -> nn_1 forward pass and weight update -> observer update law -> nn_2 forward pass and weight update -> controller update law

        Args:
            step (int): Current time step

        Returns:
            None: Updates all necessary objects in place
        """
        self.target.update_plant(step=step)

        for agent in self.agents:
            agent.measure(step=step)

        for agent in self.agents:
            agent.neighborhood_measure(step=step)

        (nn_1_global_output, nn_1_global_jacobian) = self.nn_1_jacobian_map(
            self.construct_nn_input_matrix(
                step=step,
                num_inputs=self.config.nn_1_num_inputs,
                num_agents=self.config.num_agents,
                nn_purpose="observer",
            ),
            self.nn_1_weights_dict,
        )

        # assign outputs and jacobians
        nn_1_global_output = jax.device_get(nn_1_global_output)
        nn_1_global_jacobian_dict = jax.device_get(nn_1_global_jacobian)
        nn_1_global_jacobian_matrix = self.extract_jacobian(
            num_agents=self.config.num_agents,
            global_jacobian_dict=nn_1_global_jacobian_dict,
        )
        # weight difference
        nn_1_weight_difference = np.broadcast_to(
            self.nn_1_weights_array[:, np.newaxis, :],
            (
                self.config.num_agents,
                self.config.num_agents,
                self.config.nn_1_num_weights,
            ),
        ) - np.broadcast_to(
            self.nn_1_weights_array[np.newaxis, :, :],
            (
                self.config.num_agents,
                self.config.num_agents,
                self.config.nn_1_num_weights,
            ),
        )
        # gradient corrections
        nn_1_global_observer_correction_array = np.einsum(
            "ijnp,ijp -> in", nn_1_global_jacobian_matrix, nn_1_weight_difference
        )
        nn_1_global_weight_correction_array = np.einsum(
            "ijnp -> inp", nn_1_global_jacobian_matrix
        )
        for agent in self.agents:
            # observer assignment terms
            agent.nn_1_gradient_correction = nn_1_global_observer_correction_array[
                agent.node_id, :
            ]
            # nn_1 assignment terms
            assert agent.nn_1 is not None
            agent.nn_1_output[:, step] = nn_1_global_output[agent.node_id, :]
            agent.nn_1.gradient_wrt_weights = nn_1_global_jacobian_matrix[
                agent.node_id, agent.node_id, :, :
            ]
            agent.nn_1.gradient_correction = nn_1_global_weight_correction_array[
                agent.node_id, :, :
            ]

        for agent in self.agents:
            assert agent.nn_1 is not None
            agent.update_law_1(step=step)
            self.nn_1_weights_array[agent.node_id, :] = agent.nn_1.weights

        self.nn_1_weights_dict = unflatten_all_weights(
            num_agents=self.config.num_agents,
            nn_weights_array=self.nn_1_weights_array,
            nn_schema=self.config.nn_1_schema,
        )

        for agent in self.agents:
            agent.observer(step=step)

        for agent in self.agents:
            agent.integrate(step=step)

        (nn_2_global_output, nn_2_global_jacobian) = self.nn_2_jacobian_map(
            self.construct_nn_input_matrix(
                step=step,
                num_inputs=self.config.nn_2_num_inputs,
                num_agents=self.config.num_agents,
                nn_purpose="controller",
            ),
            self.nn_2_weights_dict,
        )

        # assign outputs and jacobians
        nn_2_global_output = jax.device_get(nn_2_global_output)
        nn_2_global_jacobian_dict = jax.device_get(nn_2_global_jacobian)
        nn_2_global_jacobian_matrix = self.extract_jacobian(
            num_agents=self.config.num_agents,
            global_jacobian_dict=nn_2_global_jacobian_dict,
        )
        # weight difference
        nn_2_weight_difference = np.broadcast_to(
            self.nn_2_weights_array[:, np.newaxis, :],
            (
                self.config.num_agents,
                self.config.num_agents,
                self.config.nn_2_num_weights,
            ),
        ) - np.broadcast_to(
            self.nn_2_weights_array[np.newaxis, :, :],
            (
                self.config.num_agents,
                self.config.num_agents,
                self.config.nn_2_num_weights,
            ),
        )
        # gradient corrections
        nn_2_global_observer_correction_array = np.einsum(
            "ijnp,ijp -> in", nn_2_global_jacobian_matrix, nn_2_weight_difference
        )
        nn_2_global_weight_correction_array = np.einsum(
            "ijnp -> inp", nn_2_global_jacobian_matrix
        )

        for agent in self.agents:
            # controller assignment terms
            agent.nn_2_gradient_correction = nn_2_global_observer_correction_array[
                agent.node_id, :
            ]
            # nn_2 assignment terms
            assert agent.nn_2 is not None
            agent.nn_2_output[:, step] = nn_2_global_output[agent.node_id, :]
            agent.nn_2.gradient_wrt_weights = nn_2_global_jacobian_matrix[
                agent.node_id, agent.node_id, :, :
            ]
            agent.nn_2.gradient_correction = nn_2_global_weight_correction_array[
                agent.node_id, :, :
            ]

        for agent in self.agents:
            assert agent.nn_2 is not None
            agent.update_law_2(step=step)
            self.nn_2_weights_array[agent.node_id, :] = agent.nn_2.weights
            agent.update_plant(step=step)

        self.nn_2_weights_dict = unflatten_all_weights(
            num_agents=self.config.num_agents,
            nn_weights_array=self.nn_2_weights_array,
            nn_schema=self.config.nn_2_schema,
        )

        # store output layer weights for plotting
        self.nn_1_agent_output_weights_trajectory[:, step] = self.nn_1_weights_array[
            0, :
        ]
        self.nn_2_agent_output_weights_trajectory[:, step] = self.nn_2_weights_array[
            0, :
        ]

    def construct_nn_input_matrix(
        self,
        step: int,
        num_inputs: int,
        num_agents: int,
        nn_purpose: str,  # "observer" or "controller"
    ) -> jnp.ndarray:
        """
        Constructs (N,d) global input matrix for the NN

        Args:
            step (int): Current time step
            num_inputs (int): Number of NN inputs
            num_agents (int): Number of agents in the MAS
            nn_purpose (str): "observer" or "controller," used to determine which inputs to grab
        Returns:
            jnp.ndarray: (N,d) matrix of GNN inputs
        """

        global_input_matrix = np.zeros((num_agents, num_inputs))
        for agent in self.agents:
            if nn_purpose == "observer":
                global_input_matrix[agent.node_id, :] = np.concatenate(
                    [
                        agent.target_estimate[:, step - 1],
                        agent.target_estimate_dot[:, step - 1],
                    ]
                )
            elif nn_purpose == "controller":
                global_input_matrix[agent.node_id, :] = np.concatenate(
                    [agent.positions[:, step - 1], agent.velocities[:, step - 1]]
                )
            else:
                raise ValueError("nn_purpose must be 'controller' or 'observer")
        return jnp.asarray(global_input_matrix)

    def extract_jacobian(
        self,
        num_agents: int,
        global_jacobian_dict: dict,
    ) -> np.ndarray:
        """
        Returns (num_outputs, num_params) matricized Jacobian for use in weight update

        Args:
            num_agents (int): Number of agents in the MAS
            global_jacobian_dict (dict): Dictionary output from Jax containing Jacobian parameters
        Returns:
            np.ndarray: (num_outputs, num_params) Jacobian matrix
        """

        jacobian_list = []
        for message_passing_layer_idx in sorted(global_jacobian_dict.keys()):
            # mlp_inner always exists
            for jacobian_tensor in global_jacobian_dict[message_passing_layer_idx][
                "mlp_inner"
            ]:
                jacobian_list.append(
                    jacobian_tensor.transpose(0, 2, 1, 3, 4).reshape(
                        (
                            num_agents,
                            num_agents,
                            jacobian_tensor.shape[1],
                            jacobian_tensor.shape[3] * jacobian_tensor.shape[4],
                        ),
                        order="F",
                    )
                )
            if global_jacobian_dict[message_passing_layer_idx]["mlp_outer"] != []:
                for jacobian_tensor in global_jacobian_dict[message_passing_layer_idx][
                    "mlp_outer"
                ]:
                    jacobian_list.append(
                        jacobian_tensor.transpose(0, 2, 1, 3, 4).reshape(
                            (
                                num_agents,
                                num_agents,
                                jacobian_tensor.shape[1],
                                jacobian_tensor.shape[3] * jacobian_tensor.shape[4],
                            ),
                            order="F",
                        )
                    )

        return np.concatenate(jacobian_list, axis=3)
