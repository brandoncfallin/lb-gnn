from typing import Callable

import jax
import jax.numpy as jnp


def mlp(weights_list: list, input: jnp.ndarray) -> jnp.ndarray:
    """
    Multilayer perceptron helper function with tanh inner layer activation and linear output layer activation

    Args:
        weights_list (list): List of jnp.ndarray objects for each MLP layer of dimension (d(k) x d(k-1)+1)
        input (jnp.ndarray): MLP input vector

    Returns:
        jnp.ndarray: MLP output
    """
    layer_output = input.copy()
    for layer_idx, weight_matrix in enumerate(weights_list):
        layer_output = weight_matrix @ jnp.concatenate([layer_output, jnp.ones(1)])
        if layer_idx < len(weights_list) - 1:
            layer_output = jnp.tanh(layer_output)
    return layer_output


def dnn_forward(
    global_feature_matrix: jnp.ndarray,
    nn_weights_dict: dict,
    adjacency_matrix: jnp.ndarray,  # unused, keep for consistency with gnn_forward
    augmented_adjacency_matrix: jnp.ndarray,
    num_agents: int,
    input_dim: int,
    is_graph_valued: bool,
) -> jnp.ndarray:
    """
    Executes the DNN forward pass for a given input and weight specification

    Args:
        global_feature_matrix (jnp.ndarray): (num_agents, input_dim) array populated will features from all agents
        nn_weights_dict (dict): Dictionary of the appropriate schema with weight vals as jnp.ndarray objects
        adjacency_matrix (jnp.ndarray): placeholder for gnn_forward
        augmented_adjacency_matrix (jnp.ndarray): (num_agents, num_agents) adjacency matrix with self-loops, compute this once!
        num_agents (int): Number of agents in the network
        input_dim (int): Shape of DNN input
        output_dim (int): Shape of DNN output
        is_graph_valued (bool): Indicates if we are whacky stacking or not
    Returns:
        jnp.ndarray: (num_agents, output_dim) DNN output for a given input
    """

    if is_graph_valued:
        adj_kron = jnp.kron(augmented_adjacency_matrix, jnp.ones((1, input_dim)))
        feat_kron = jnp.kron(
            global_feature_matrix.reshape((1, num_agents * input_dim)),
            jnp.ones((num_agents, 1)),
        )
        global_feature_matrix = (
            adj_kron * feat_kron
        )  # (num_agents, num_agents * input_dim)

    weights_list_batched = nn_weights_dict[0]["mlp_inner"]  # all dnn weights in here

    # vmap over agent indices
    batched_mlp = jax.vmap(mlp)
    return batched_mlp(weights_list=weights_list_batched, input=global_feature_matrix)


def gnn_forward(
    global_feature_matrix: jnp.ndarray,
    nn_weights_dict: dict,  # unused, keep for consistency with dnn_forward
    adjacency_matrix: jnp.ndarray,
    augmented_adjacency_matrix: jnp.ndarray,  # unused, keep for consistency with dnn_forward
    num_agents: int,  # unused, keep for consistency with dnn_forward
    input_dim: int,  # unused, keep for consistency with dnn_forward
    is_graph_valued: bool,  # unused, keep for consistency with dnn_forward
) -> jnp.ndarray:
    """
    Performs GNN forward pass and returns output

    Args:
        global_feature_matrix (jnp.ndarray): (N,d) input features
        nn_weights_dict (dict): Dict of nn weights in correct schema
        adjacency_matrix (jnp.ndarray): (N,N) array indicating connections between agents, no self-loops
        augmented_adjacency_matrix (jnp.ndarray): Unused; for parity with dnn_forward
        num_agents (int): Unused; for parity with dnn_forward
        input_dim (int): Unused; for parity with dnn_forward
        is_graph_valued (bool): Unused; for parity with dnn_forward

    Returns:
        jnp.ndarray: GNN output
    """

    def fixed_weight_mlp(weights_list: list, global_feature_matrix: jnp.ndarray):
        return jax.vmap(lambda agent_j_input: mlp(weights_list, agent_j_input))(
            global_feature_matrix
        )

    layer_output = global_feature_matrix.copy()
    batched_inner = jax.vmap(fixed_weight_mlp, in_axes=(0, None))
    batched_outer = jax.vmap(
        lambda agent_i_weights, agent_i_input: mlp(agent_i_weights, agent_i_input),
        in_axes=(0, 0),
    )
    for message_passing_layer_idx in nn_weights_dict.keys():
        aggregate_here = batched_inner(
            nn_weights_dict[message_passing_layer_idx]["mlp_inner"], layer_output
        )
        # jax.debug.print("aggregate_here {x}", x=aggregate_here.shape)
        aggregate_here = aggregate_here * adjacency_matrix[:, :, None]
        aggregated = aggregate_here.sum(axis=1)
        # jax.debug.print("aggregated {x}", x=aggregated.shape)
        outer_input = jnp.concatenate([layer_output, aggregated], axis=1)
        # jax.debug.print("outer_input {x}", x=outer_input.shape)
        layer_output = batched_outer(
            nn_weights_dict[message_passing_layer_idx]["mlp_outer"], outer_input
        )
        # jax.debug.print("layer_output {x}", x=layer_output.shape)
    return layer_output


# build_jacobian_map
# to be called once for each NN, globally (i.e., \Phi_1 and \Phi_2)
# jit compiles fwd pass for speed
# returns a map from (features, weights_dict) -> (outputs, jacobian)
def build_jacobian_map(
    forward_pass: Callable,
    adjacency_matrix: jnp.ndarray,
    augmented_adjacency_matrix: jnp.ndarray,
    num_agents: int,
    input_dim: int,
    is_graph_valued: bool,
) -> Callable:
    """
    Assembles function which returns NN Jacobian dict for an input feature matrix and weights dictionary

    Args:
        forward_pass (Callable): Expects dnn_forward() or gnn_forward()
        adjacency_matrix (jnp.ndarray): (N,N) array indicating connections between agents, NO self-loops
        augmented_adjacency_matrix (jnp.ndarray): (N,N) array indicating connections between agents, WITH self-loops
        num_agents (int): Number of agents in MAS
        input_dim (int): Number of inputs at each agent
        is_graph_valued (bool): Check if assembling graph-wide values for DNN

    Returns:
        Callable: Jacobian map function, uses Jax
    """

    @jax.jit
    def jacobian_map(global_feature_matrix: jnp.ndarray, nn_weights_dict: jnp.ndarray):
        global_output = forward_pass(
            global_feature_matrix=global_feature_matrix,
            nn_weights_dict=nn_weights_dict,
            adjacency_matrix=adjacency_matrix,
            augmented_adjacency_matrix=augmented_adjacency_matrix,
            num_agents=num_agents,
            input_dim=input_dim,
            is_graph_valued=is_graph_valued,
        )
        global_jacobian = jax.jacrev(
            lambda weights: forward_pass(
                global_feature_matrix=global_feature_matrix,
                nn_weights_dict=weights,
                adjacency_matrix=adjacency_matrix,
                augmented_adjacency_matrix=augmented_adjacency_matrix,
                num_agents=num_agents,
                input_dim=input_dim,
                is_graph_valued=is_graph_valued,
            )
        )(nn_weights_dict)
        return global_output, global_jacobian

    return jacobian_map
