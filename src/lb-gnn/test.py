import jax.numpy as jnp


def local_neighborhood_mlp(
    all_features, all_weights_inner, all_weights_outer, adj_matrix
):
    """
    Vectorized computation of neighborhood MLP for all agents simultaneously.

    Args:
        all_features: (N, F) - features for each node
        all_weights_inner: (N, H, F+1) - inner layer weights for each agent
        all_weights_outer: (N, Out, H+1) - outer layer weights for each agent
        adj_matrix: (N, N) - adjacency matrix

    Returns:
        neighbor_sum: (N, Out) - aggregated neighborhood features for each agent
    """
    N, _ = all_features.shape

    # 1. Augment features with bias: (N, F+1)
    X_padded = jnp.concatenate([all_features, jnp.ones((N, 1))], axis=-1)

    # 2. Inner layer: Each agent i processes each node j's features
    # We want: for each agent i, for each node j, compute hidden_ij = W_inner_i @ X_j
    # X_padded: (N, F+1) -> node features (j indexes this)
    # all_weights_inner: (N, H, F+1) -> agent-specific weights (i indexes agents)
    # Result should be: (N_agents, N_nodes, H)

    # Compute: hidden[i,j,h] = sum_f(W_inner[i,h,f] * X[j,f])
    hidden = jnp.einsum("jf,ihf->ijh", X_padded, all_weights_inner)
    # Shape: (N_agents, N_nodes, H)

    # 3. Apply activation
    hidden_act = jnp.tanh(hidden)

    # 4. Augment hidden layer with bias: (N, N, H+1)
    hidden_padded = jnp.concatenate(
        [hidden_act, jnp.ones((*hidden_act.shape[:2], 1))], axis=-1
    )

    # 5. Outer layer: Each agent i processes its perception of node j
    # hidden_padded: (N_agents, N_nodes, H+1)
    # all_weights_outer: (N_agents, Out, H+1)
    # Result: (N_agents, N_nodes, Out)
    perceptions = jnp.einsum("ijh,ioh->ijo", hidden_padded, all_weights_outer)
    # Shape: (N_agents, N_nodes, Out)

    # 6. Aggregate over neighbors using adjacency matrix
    # adj_matrix[i,j] = 1 if j is a neighbor of i
    # We want: output[i,o] = sum_j(adj[i,j] * perceptions[i,j,o])
    neighbor_sum = jnp.einsum("ij,ijo->io", adj_matrix, perceptions)
    # Shape: (N_agents, Out)

    return neighbor_sum


def vectorized_forward_layer(
    all_node_features: jnp.ndarray,  # (N, F)
    adjacency_matrix: jnp.ndarray,
    all_W_layer,
    all_X_layer,
    all_V_layer,
    all_U_layer,
) -> jnp.ndarray:
    """
    Vectorized forward pass for a single layer across all agents.
    """
    N = all_node_features.shape[0]

    # First MLP: W(X(...))
    aggregated_neighbor_features = local_neighborhood_mlp(
        all_features=all_node_features,
        all_weights_inner=all_X_layer,
        all_weights_outer=all_W_layer,
        adj_matrix=adjacency_matrix,
    )  # (N, hidden_dim)

    # Concatenate own features with aggregated neighbor features
    combined_features = jnp.concatenate(
        [all_node_features, aggregated_neighbor_features], axis=-1
    )  # (N, 2*input_dim)

    # Second MLP: U(V(...))
    output_features = local_neighborhood_mlp(
        all_features=combined_features,
        all_weights_inner=all_V_layer,
        all_weights_outer=all_U_layer,
        adj_matrix=jnp.eye(N),  # Only process self, no aggregation
    )  # (N, output_dim)

    return output_features


def jax_forward_pass(
    adjacency_matrix: jnp.ndarray,
    input_matrix: jnp.ndarray,
    U_list,
    V_list,
    W_list,
    X_list,
    active_node: int,
    num_layers: int,
    neighbors_by_layer: dict[int, jnp.ndarray],
    hidden_activation: str,
    output_activation: str,
) -> jnp.ndarray:
    previous_layer_output = input_matrix.copy()
    # perform forward pass
    for layer in range(num_layers):
        layer_output = vectorized_forward_layer(
            all_node_features=previous_layer_output,
            adjacency_matrix=adjacency_matrix,
            all_W_layer=W_list[layer],
            all_X_layer=X_list[layer],
            all_V_layer=V_list[layer],
            all_U_layer=U_list[layer],
        )

        previous_layer_output = layer_output.copy()

    return layer_output[active_node]


def prepare_weights_dict(
    num_agents: int, num_layers: int, weights_dict: dict
) -> tuple[list, list, list, list]:
    U_list = []
    V_list = []
    W_list = []
    X_list = []

    for layer in range(num_layers):
        U_placeholder = []
        V_placeholder = []
        W_placeholder = []
        X_placeholder = []
        for agent in range(num_agents):
            U_placeholder.append(weights_dict[(agent, layer, "U")])
            V_placeholder.append(weights_dict[(agent, layer, "V")])
            W_placeholder.append(weights_dict[(agent, layer, "W")])
            X_placeholder.append(weights_dict[(agent, layer, "X")])

        # stack into (num_agents, output_size, input_size)
        U_list.append(jnp.stack(U_placeholder))
        V_list.append(jnp.stack(V_placeholder))
        W_list.append(jnp.stack(W_placeholder))
        X_list.append(jnp.stack(X_placeholder))
    return U_list, V_list, W_list, X_list


def jacfwd_wrt_active_node_weights(
    self,
    adjacency_matrix: jnp.ndarray,
    input_matrix: jnp.ndarray,
    U_list,
    V_list,
    W_list,
    X_list,
    weight_deriv_node: int,
    num_layers: int,
    hidden_activation: str,
    output_activation: str,
) -> jnp.ndarray:
    """
    Computes the Jacobian of the active node's output with respect to
    the weights of weight_deriv_node.

    Returns: jnp.ndarray of shape (num_outputs, total_weight_dim)
    """

    # 1. Define the local Pytree for the node being differentiated.
    # Order here matters for the eventual flattening.
    local_weights_template = {
        "U": [L[weight_deriv_node] for L in U_list],
        "V": [L[weight_deriv_node] for L in V_list],
        "W": [L[weight_deriv_node] for L in W_list],
        "X": [L[weight_deriv_node] for L in X_list],
    }

    def forward_wrapper(local_w_pytree):
        # 2. Inject the 'differentiable' weights into the vectorized stacks
        # Using .at().set() inside JAX is efficient for JIT/autodiff.
        U_new = [
            U.at[weight_deriv_node].set(local_w_pytree["U"][i])
            for i, U in enumerate(U_list)
        ]
        V_new = [
            V.at[weight_deriv_node].set(local_w_pytree["V"][i])
            for i, V in enumerate(V_list)
        ]
        W_new = [
            W.at[weight_deriv_node].set(local_w_pytree["W"][i])
            for i, W in enumerate(W_list)
        ]
        X_new = [
            X.at[weight_deriv_node].set(local_w_pytree["X"][i])
            for i, X in enumerate(X_list)
        ]

        return jax_forward_pass(
            adjacency_matrix=adjacency_matrix,
            input_matrix=input_matrix,
            U_list=U_new,
            V_list=V_new,
            W_list=W_new,
            X_list=X_new,
            active_node=self.node_id,  # Target agent
            num_layers=num_layers,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )

    # 3. Compute the Jacobian with respect to the Pytree
    # If the output of forward_pass is (num_outputs,),
    # J_pytree will be a dict of lists with shapes (num_outputs, weight_shape)
    J_pytree = jacfwd(forward_wrapper)(local_weights_template)

    # 4. Flatten the Jacobian while maintaining the output dimension
    # We want (num_outputs, flat_weight_dim).
    # JAX's jacfwd output can be tricky to flatten manually.

    def flatten_jacobian(j_tree):
        # We need to concatenate all Jacobian components along the weight axis (axis=-1)
        # components will be (num_outputs, weight_elements)
        flat_components = []
        # Ensure we iterate in the exact same order: U, V, W, X
        for key in ["U", "V", "W", "X"]:
            for layer_grad in j_tree[key]:
                # Reshape from (num_outputs, dim1, dim2) to (num_outputs, -1)
                flat_components.append(layer_grad.reshape(layer_grad.shape[0], -1))
        return jnp.concatenate(flat_components, axis=-1)

    return flatten_jacobian(J_pytree)
