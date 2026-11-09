import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.linalg import block_diag

# weights pytree schema
# 2 types of weights: mlp_inner and mlp_outer
# recall that gnn arch is of the form
# mlp_outer(x_{i,k-1} + \sum_{j \in N_1} mlp_inner(x_{j,k-1}))

# all dnn weights are placed into mlp_inner
# layers go in ascending order (inside out)
# construct dict of dicts of the form
# message_passing_layer_0 = {
#   'mlp_inner': [(H, F+1),(F,H+1)], # single hidden layer + output layer F -> H -> F
#   'mlp_outer': [(H, 2*F+1),(O,H+1)], # 2*F -> H -> O
# }
# nn_schema = {0: message_passing_layer_0, ..., k-1: message_passing_layer_k-1}


# dnn, gnn weight initialization
def initialize_weights(
    num_agents: int,
    nn_schema: dict,
    rng: Generator,
    same_at_each_agent: bool,
) -> dict:
    """
    Initialize weights for GNN or DNN using Glorot initialization

    Args:
        num_agents (int): The number of agents in the network
        nn_schema (dict): Dictionary with keys of message passing layers indexed from 0 and values of dictionaries describing the weight structure at that layer
        rng (Generator): numpy rng with a specific seed
        same_at_each_agent (bool): Indicates if weights should be the same at each agent

    Returns:
        dict: Dictionary with keys of message passing layers indexed from 0 and values of dictionaries containing weights for each layer
    """

    # glorot initialization
    # make blank dict to populate
    nn_weights = {}
    for layer_index in sorted(nn_schema.keys()):
        message_passing_layer_schema = nn_schema[layer_index]  # grab current schema
        nn_weights[layer_index] = {"mlp_inner": [], "mlp_outer": []}  # blank layer dict
        # generate mlp_inner_weights (should always be present)
        for size_tuple in message_passing_layer_schema["mlp_inner"]:
            # do not incl bias in glorot
            input_dim = size_tuple[1] - 1
            output_dim = size_tuple[0]
            glorot = np.sqrt(6 / (input_dim + output_dim))
            if same_at_each_agent:
                weight_matrix = rng.uniform(low=-glorot, high=glorot, size=size_tuple)
                nn_weights[layer_index]["mlp_inner"].append(
                    jnp.tile(weight_matrix, (num_agents, 1, 1))
                )
            else:
                global_weight_matrix = rng.uniform(
                    low=-glorot, high=glorot, size=(num_agents, *size_tuple)
                )
                nn_weights[layer_index]["mlp_inner"].append(
                    jnp.asarray(global_weight_matrix)
                )
        # generate mlp_outer_weights (may not be present for DNN)
        if message_passing_layer_schema["mlp_outer"] != []:
            for size_tuple in message_passing_layer_schema["mlp_outer"]:
                # do not incl bias in glorot
                input_dim = size_tuple[1] - 1
                output_dim = size_tuple[0]
                glorot = np.sqrt(6 / (input_dim + output_dim))
                if same_at_each_agent:
                    weight_matrix = rng.uniform(
                        low=-glorot, high=glorot, size=size_tuple
                    )
                    # weight_matrix = rng.normal(loc=0.0, scale=0.3, size=size_tuple)
                    nn_weights[layer_index]["mlp_outer"].append(
                        jnp.tile(weight_matrix, (num_agents, 1, 1))
                    )
                else:
                    global_weight_matrix = rng.uniform(
                        low=-glorot, high=glorot, size=(num_agents, *size_tuple)
                    )
                    nn_weights[layer_index]["mlp_outer"].append(
                        jnp.asarray(global_weight_matrix)
                    )
    return nn_weights


# projection operator
def proj(
    theta_hat_dot: NDArray[np.float64],
    theta_hat: NDArray[np.float64],
    gamma_matrix: NDArray[np.float64],
    theta_bar: float,
) -> NDArray[np.float64]:
    """
    Compute the safe weight update given upper and lower bounds

    Note:
        Projection operator defined in "A Sufficiently Smooth Projection Operator", Cai (2006)

    Args:
        theta_hat_dot (np.ndarray): The proposed update direction for the parameter estimate.
        theta_hat (np.ndarray): The current parameter estimate.
        gamma_matrix (np.ndarray): The adaptation gain matrix.
        theta_bar (float): The upper bound on the parameter estimate norm.

    Returns:
        np.ndarray: The projected update direction for the parameter estimate.
    """

    theta_hat = np.asarray(theta_hat, dtype=np.float64).reshape(-1)
    theta_hat_dot = np.asarray(theta_hat_dot, dtype=np.float64).reshape(-1)
    gamma_matrix = np.asarray(gamma_matrix, dtype=np.float64)

    if (theta_hat.T @ theta_hat) >= theta_bar**2:
        is_on_or_outside_boundary = True
    else:
        is_on_or_outside_boundary = False
    outward_component = theta_hat.T @ theta_hat_dot
    if outward_component > 0:
        is_moving_outward = True
    else:
        is_moving_outward = False

    if is_on_or_outside_boundary and is_moving_outward:
        denominator = theta_hat.T @ gamma_matrix @ theta_hat
        const = outward_component / denominator
        correction_term = const * (gamma_matrix @ theta_hat)
        theta_hat_dot_safe: NDArray[np.float64] = theta_hat_dot - correction_term
    else:
        theta_hat_dot_safe = theta_hat_dot

    return theta_hat_dot_safe


# build_gamma_matrix (for update law), run once
def build_gamma_matrix(gamma_params: dict, nn_schema: dict) -> NDArray:
    """
    Constructs gamma matrix for adaptive update law based on input schemas. Same for every agent, only needs to be computed once.

    Args:
        gamma_params (dict): Dict with keys of "mlp_inner" and "mlp_outer" with corresponding scalar values
        nn_schema (dict): You know the one

    Returns:
        np.ndarray: (num_weights,num_weights) matrix
    """

    gamma_matrix_list = []
    for message_passing_layer_index in sorted(nn_schema.keys()):
        # make for mlp_inner (always exists)
        for size_tuple in nn_schema[message_passing_layer_index]["mlp_inner"]:
            gamma_matrix_list.append(
                gamma_params["mlp_inner"] * np.eye(size_tuple[0] * size_tuple[1])
            )

        # make for mlp_outer (may not exist for dnn)
        if nn_schema[message_passing_layer_index]["mlp_outer"] != []:
            for size_tuple in nn_schema[message_passing_layer_index]["mlp_outer"]:
                gamma_matrix_list.append(
                    gamma_params["mlp_outer"] * np.eye(size_tuple[0] * size_tuple[1])
                )

    return block_diag(*gamma_matrix_list)


# convert from dict to numpy array
def flatten_all_weights(
    num_agents: int,
    num_weights: int,
    nn_weights_dict: dict,
) -> NDArray:
    """
    Turns the nn_weights_dict into a numpy array for application in the update law

    Args:
        num_agents (int): Number of agents in the network
        num_weights (int): Total number of parameters at each node, calculate this before hand
        nn_weights_dict (dict): Dictionary with keys of message passing layers and values of dictionaries containing MLP weights

    Returns:
        np.ndarray: (N,num_weights) numpy array
    """

    # convert to numpy
    nn_weights_dict = jax.device_get(nn_weights_dict)

    # loop over weights pytree. can this be done in parallel?
    weights_list = []
    for message_passing_layer in sorted(nn_weights_dict.keys()):
        # mlp_inner (always exists)
        for weight_tensor in nn_weights_dict[message_passing_layer]["mlp_inner"]:
            # recall this is a (N, output_dim, input_dim) matrix
            weights_list.append(
                weight_tensor.reshape(
                    (num_agents, weight_tensor.shape[1] * weight_tensor.shape[2]),
                    order="F",
                )
            )
        # mlp_outer (does not exist for dnn)
        if nn_weights_dict[message_passing_layer]["mlp_outer"] != []:
            for weight_tensor in nn_weights_dict[message_passing_layer]["mlp_outer"]:
                # recall this is a (N, output_dim, input_dim) matrix
                weights_list.append(
                    weight_tensor.reshape(
                        (num_agents, weight_tensor.shape[1] * weight_tensor.shape[2]),
                        order="F",
                    )
                )

    nn_weights_array = np.concatenate(weights_list, axis=1)
    assert nn_weights_array.shape[1] == num_weights, (
        f"Expected {num_weights}, got {nn_weights_array.shape[1]}"
    )
    return nn_weights_array


# unflatten_all
# input is N, param_dim numpy_array
# returns nn_weights dict with jnp vals
# also runs every time step, so needs to be FAST
def unflatten_all_weights(
    num_agents: int, nn_weights_array: NDArray, nn_schema: dict
) -> dict:
    """
    Converts a numpy array of flattened weights into jax pytree

    Args:
        num_agents (int): Number of agents in the network
        nn_weights_array (np.ndarray): (N,num_weights) numpy array of weights
        nn_schema (dict): Dictionary of the weight specification
    Returns:
        dict: Dictionary containing weight values for each message passing layer
    """

    nn_weights_jax = jnp.asarray(nn_weights_array)
    nn_weights_dict = {}
    idx = 0
    for message_passing_layer_index in sorted(nn_schema.keys()):
        # make entry for this layer
        nn_weights_dict[message_passing_layer_index] = {
            "mlp_inner": [],
            "mlp_outer": [],
        }
        # mlp_inner (always exists)
        for mlp_layer_size in nn_schema[message_passing_layer_index]["mlp_inner"]:
            current_matrix = nn_weights_jax[
                :, idx : idx + mlp_layer_size[0] * mlp_layer_size[1]
            ]
            nn_weights_dict[message_passing_layer_index]["mlp_inner"].append(
                current_matrix.reshape(
                    (num_agents, mlp_layer_size[0], mlp_layer_size[1]), order="F"
                )
            )
            idx += mlp_layer_size[0] * mlp_layer_size[1]
        # mlp_outer (does not exist for dnn)
        if nn_schema[message_passing_layer_index]["mlp_outer"] != []:
            for mlp_layer_size in nn_schema[message_passing_layer_index]["mlp_outer"]:
                current_matrix = nn_weights_jax[
                    :, idx : idx + mlp_layer_size[0] * mlp_layer_size[1]
                ]
                nn_weights_dict[message_passing_layer_index]["mlp_outer"].append(
                    current_matrix.reshape(
                        (num_agents, mlp_layer_size[0], mlp_layer_size[1]), order="F"
                    )
                )
                idx += mlp_layer_size[0] * mlp_layer_size[1]

    return nn_weights_dict
