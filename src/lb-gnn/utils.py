import multiprocessing
import os

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from jax import grad, jacfwd, jacrev
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike, DTypeLike
from numpy.typing import NDArray
from setup import Config
from simulation import Simulation


# dict update
def update_dictionaries(dest: dict, src: dict) -> None:
    """
    Helper function to update one dict with another's values

    Args:
        dest: The destination dictionary to be updated
        src: The source dictionary whose values will be used to update the destination
    Returns:
        None: The destination dictionary is updated in place
    """
    dest.update(src)


# for kronecker with identity on the left
def kron_eye_A(arr: NDArray, num: int) -> NDArray:
    """
    Fast Kronecker product of an identity matrix with another matrix (i.e. I ⊗ A)

    Args:
        arr: The input matrix A
        num: The size of the identity matrix I
    Returns:
        result: The Kronecker product I ⊗ A
    """
    rows, cols = arr.shape
    result = np.zeros((num, rows, num, cols), dtype=arr.dtype)
    diag = np.einsum("ijik->ijk", result)
    diag[:] = arr
    return result.reshape(rows * num, cols * num)


# for kronecker with identity on the right
def kron_A_eye(arr: NDArray, num: int) -> NDArray:
    """
    Fast Kronecker product of matrix with an identity matrix (i.e. A ⊗ I)

    Args:
        arr: The input matrix A
        num: The size of the identity matrix I
    Returns:
        result: The Kronecker product A ⊗ I
    """
    rows, cols = arr.shape
    result = np.zeros((rows, num, cols, num), dtype=arr.dtype)
    r = np.arange(num)
    result[:, r, :, r] = arr
    result.shape = (rows * num, cols * num)
    return result


# make pinning vector
def make_pin(num_agents: int, pinned_agents: list[int]) -> list[int]:
    """
    Makes the pinning vector based on the agents linked to the target in the config.
    Agents that are linked to the target are pinned (1), while those that are not
    linked are unpinned (0).

    Args:
        num_agents (int): Total number of agents
        pinned_agents (list[int]): List of agent indices that are linked to the target
    Returns:
        pinning_vector: A list of 0s and 1s indicating which agents are pinned (1)
                        and which are not (0)
    """

    pinning_vector = []
    for agent in range(num_agents):
        if agent in pinned_agents:
            pinning_vector.append(1)
        else:
            pinning_vector.append(0)
    return pinning_vector


# make adjacency matrix
def make_adjacency_matrix(
    num_agents: int, edge_set: list[tuple[int, int]]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Makes adjacency matrix [N x N] and augmented adjacency matrix [N x N] based on the
    edge set in the simulation

    Args:
        num_agents (int): Total number of agents
        edge_set (list[tuple[int, int]]): List of edges represented as tuples (i, j)

    Returns:
        adjacency_matrix: A [N x N] matrix where entry (i, j) is 1 if there is an edge
                          from agent i to agent j, and 0 otherwise.
        augmented_adjacency_matrix: A [N x N] matrix where entry (i, j) is 1 if there
                                   is an edge from agent i to agent j or if i == j,
                                   and 0 otherwise.
    """

    adjacency_matrix = np.zeros((num_agents, num_agents))
    augmented_adjacency_matrix = np.zeros((num_agents, num_agents))
    for i, j in edge_set:
        adjacency_matrix[i, j] = 1
        augmented_adjacency_matrix[i, j] = 1
    for i in range(num_agents):
        augmented_adjacency_matrix[i, i] = 1
    return adjacency_matrix, augmented_adjacency_matrix


# make edge set
def make_edge_set(num_agents: int, topology: str) -> list[tuple[int, int]]:
    """
    Given a desired graph topology and number of agents, create the corresponding
    edge set

    Args:
        num_agents (int): Total number of agents
        topology (str): Desired graph topology ("ring", "star", "path", "complete", "acyclic")
    Returns:
        edge_set (list[tuple[int, int]]): List of edges represented as tuples (i, j)
    """

    edge_set = set()
    if topology == "ring":
        for i in range(num_agents):
            edge_set.add((i, (i + 1) % num_agents))
            edge_set.add(((i + 1) % num_agents, i))
    elif topology == "star":
        for i in range(num_agents):
            edge_set.add((2, i))
            edge_set.add((i, 2))
    elif topology == "path":
        for i in range(num_agents - 1):
            edge_set.add((i, i + 1))
            edge_set.add((i + 1, i))
    elif topology == "complete":
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if i != j:
                    edge_set.add((i, j))
                    edge_set.add((j, i))
    elif topology == "acyclic":
        nodes_left = list(range(num_agents))
        nodes_left.remove(1)
        current_level = [1]
        children_count = 2

        while nodes_left and current_level:
            next_level = []
            total_children = min(len(nodes_left), len(current_level) * children_count)
            children_per_parent = total_children // len(current_level)
            remaining = total_children % len(current_level)

            for parent in current_level:
                parent_children = children_per_parent + (1 if remaining > 0 else 0)
                remaining = max(0, remaining - 1)

                for _ in range(parent_children):
                    if nodes_left:
                        child = nodes_left.pop(0)
                        next_level.append(child)
                        edge_set.add((parent, child))
                        edge_set.add((child, parent))

            current_level = next_level
            children_count += 1

    return list(edge_set)


def kronecker_delta(a: int, b: int) -> int:
    """
    Kronecker delta function

    Args:
        a: First integer
        b: Second integer
    Returns:
        1 if a == b, else 0
    """
    if a == b:
        return 1
    else:
        return 0


def rnd_result(input_vals: list[np.float64]) -> list[float]:
    """
    Rounds each float in a list to 3 decimal places (this is me being lazy)

    Args:
        input_vals (list[np.float64]): List of inputs to be rounded
    Returns:
        list[float]: Rounded values
    """
    return [round(float(val), 3) for val in input_vals]


def get_k_hop_neighbors(
    node_index: int, k: int, adjacency_matrix: jnp.ndarray
) -> jnp.ndarray:
    """
    Get the k-hop neighbors of a given node in a graph that is represented by adjacency
    matrix.

    Args:
        node_index (int): Active node index
        k (int): Number of hops
        adjacency_matrix (jnp.ndarray): Adjacency matrix [N x N] representing the graph
    Returns:
        jnp.ndarray: Indices of k-hop neighbors
    """

    if k == 0:
        return jnp.array([node_index])
    elif k == 1:
        reachability_matrix = jnp.linalg.matrix_power(adjacency_matrix, k)
        k_hop_neighbors = jnp.where(reachability_matrix[node_index] > 0)[0]
        k_hop_neighbors = jnp.append(k_hop_neighbors, node_index)
    else:
        k_hop_neighbors = jnp.array([])
        for hop in range(1, k + 1):
            reachability_matrix = jnp.linalg.matrix_power(adjacency_matrix, hop)
            neighbors_at_hop = jnp.where(reachability_matrix[node_index] > 0)[0]
            k_hop_neighbors = jnp.union1d(k_hop_neighbors, neighbors_at_hop)

    return k_hop_neighbors.astype(jnp.int32)


def get_all_k_hop_neighbors(
    node_index: int, adjacency_matrix: jnp.ndarray, num_layers: int
) -> dict[int, jnp.ndarray]:
    """
    Helper function which runs `get_k_hop_neighbors()` for each GNN layer of a single agent

    Args:
        node_index (int): Active node index
        adjacency_matrix (jnp.ndarray): Adjacency matrix [N x N] representing the graph
        num_layers (int): Number of message-passing layers
    """

    neighbors_by_layer = {}
    for layer in range(num_layers + 1):
        k = num_layers - layer - 1
        neighbors_by_layer[layer] = get_k_hop_neighbors(
            node_index=node_index, k=k, adjacency_matrix=adjacency_matrix
        )
    return neighbors_by_layer
