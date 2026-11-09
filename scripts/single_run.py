import copy
import os

import numpy as np

from lb_gnn.results import pickle_results
from lb_gnn.setup import Config
from lb_gnn.simulation import Simulation

if __name__ == "__main__":
    # env vars for running many sims
    topology = os.getenv("TOPOLOGY", "ring")
    trial_number = int(os.getenv("TRIAL_NUMBER", "0"))
    num_agents = int(os.getenv("NUM_AGENTS", "6"))

    # run variations
    run_dnn = True
    run_gnn = True
    # save results
    save_results = True
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    # global settings
    global_config = Config(
        # global settings
        trial_number=trial_number,
        nn_1_type="placeholder",
        nn_2_type="placeholder",
        dt=0.005,
        tf=30,
        num_agents=num_agents,
        topology=topology,  # ring, star, path, complete, acyclic
        pinned_agents=[0, 2, 4],  # which agents can measure target states
        # gains
        observer_gain=25,
        controller_gain=25.5,
        nn_1_forgetting_factor=0.005,
        nn_2_forgetting_factor=0.002,
        nn_1_citw=0.005,
        nn_2_citw=0.002,
        alpha_1=1.85,
        alpha_2=0.625,
        # target agent settings
        target_init_pos=np.array([-1, 2, 1]),
        target_init_vel=np.array([0.5, 0, -0.2]),
        # global neural network settings
        num_neurons_per_layer=8,
        num_mlp_layers=2,
        num_message_passing_layers=2,  # placeholder, update for each architecture
        same_weights_at_each_agent=False,
        # neural network 1 settings (for observer)
        nn_1_num_inputs=6,
        nn_1_num_outputs=3,
        nn_1_theta_bar=10,
        nn_1_gamma_params={"mlp_inner": 20.15, "mlp_outer": 20.15},
        # neural network 2 settings (for controller)
        nn_2_num_inputs=6,
        nn_2_num_outputs=3,
        nn_2_theta_bar=10,
        nn_2_gamma_params={"mlp_inner": 22.85, "mlp_outer": 22.85},
        # rng
        weight_seed=None,  # None for random at each run
    )

    # dnn setup
    dnn_config = copy.deepcopy(global_config)
    dnn_config.nn_1_type = "dnn"
    dnn_config.nn_2_type = "dnn"
    dnn_config.num_message_passing_layers = 1
    dnn_config.num_mlp_layers = 8
    dnn_config.num_neurons_per_layer = 8
    dnn_config.rng = np.random.default_rng(seed=global_config.weight_seed)
    dnn_config.initialize_nn_schemas()
    dnn_simulation = Simulation(dnn_config)
    dnn_simulation.label = "DNN"

    # gnn setup
    gnn_config = copy.deepcopy(global_config)
    gnn_config.nn_1_type = "gnn"
    gnn_config.nn_2_type = "gnn"
    gnn_config.rng = np.random.default_rng(seed=global_config.weight_seed)
    gnn_config.initialize_nn_schemas()
    gnn_simulation = Simulation(gnn_config)
    gnn_simulation.label = "GNN"

    # check which simulations to run
    results_list = [dnn_simulation, gnn_simulation]
    results_list_bool = [run_dnn, run_gnn]
    results_list = [
        results_list[i] for i in range(len(results_list)) if results_list_bool[i]
    ]

    if len(results_list) != 0:
        # run indicated sims
        for simulation in results_list:
            simulation.run()
        # pickle results
        if save_results:
            pickle_results(results_list, data_dir)
            print(f"Results saved for {topology} topology.")
