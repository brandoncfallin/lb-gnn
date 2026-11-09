import os
import pickle

import numpy as np

from .simulation import Simulation


def pickle_results(simulation_list: list[Simulation], data_dir: str) -> None:
    """
    Saves the following quantities using pickle into the /data directory:
        time, target positions, target velocity, agent positions, agent velocities, agent nn outputs, agent nn weights

    Args:
        simulation_list (list[Simulation]): List of complete Simulation objects
        data_dir (str): Output directory where .pkl file is saved
    Returns:
        None
    """

    # pickle the following quantities
    # time, target positions, target velocity, agent positions, agent velocities, agent nn outputs, agent nn weights

    # create results directory
    dirname = os.path.dirname(__file__)
    results_dir = os.path.join(dirname, data_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for simulation in simulation_list:
        results = {
            "label": simulation.label,
            "trial_number": simulation.trial_number,
            "time": np.linspace(
                0,
                simulation.config.tf,
                int(simulation.config.tf / simulation.config.dt),
            ),
            "target_positions": simulation.target.positions,
            "target_velocities": simulation.target.velocities,
            "target_accelerations": simulation.target.accelerations,
            "agent_inter": [agent.interaction_dynamics for agent in simulation.agents],
            "agent_positions": [agent.positions for agent in simulation.agents],
            "agent_velocities": [agent.velocities for agent in simulation.agents],
            "agent_control_effort": [
                agent.control_input for agent in simulation.agents
            ],
            "agent_nn_output_1": [agent.nn_1_output for agent in simulation.agents],
            "agent_nn_output_2": [agent.nn_2_output for agent in simulation.agents],
        }

        # save results
        filename = os.path.join(
            results_dir,
            f"{simulation.config.topology}_{simulation.config.nn_1_type}_{simulation.config.nn_2_type}_{simulation.config.num_agents}agents_{simulation.config.tf}s_trial{simulation.config.trial_number}.pkl",
        )
        with open(filename, "wb") as f:
            pickle.dump(results, f)
