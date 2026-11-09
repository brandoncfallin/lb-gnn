import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


def import_data(filename: str) -> dict:
    """
    Load pickle file located at the specified path

    Args:
        filename (str): File path for a single run's data

    Returns:
        dict: Data dictionary from the given run
    """

    # get data path
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base, "data", filename)
    # open file with pickle
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def import_topology(topology: str, trial_number: Optional[int] = None) -> list[dict]:
    """
    Import all files with given topology in data directory

    Args:
        topology (str): Communication topology, "path", "ring", "star", "complete", "acyclic"
        trial_number (int): Trial number. If specified, will only import the single trial
    Returns:
        list[dict]: List of data dictionaries for a given communication topology
    """

    # use import_data function to import data
    # return list of data
    data_list = []
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for file in os.listdir(os.path.join(base, "data")):
        if file.startswith(topology):
            if trial_number is None:
                data_list.append(import_data(file))
            elif file.endswith(f"trial{trial_number}.pkl"):
                data_list.append(import_data(file))
    return data_list


def calculate_params(data: dict) -> dict:
    """
    Computes relevant error statistics given an input data dictionary

    Args:
        data (dict): Data dictionary for a single run
    Returns:
        dict: Dictionary of error statistics for a single run
    """

    # get data label
    label = data["label"]

    trial_number = data["trial_number"]

    # calculate RMS position tracking error
    positions = []
    for i in range(len(data["agent_positions"])):
        positions.append(0)
        positions[i] = root_mean_squared_error(
            data["agent_positions"][i][:, :], data["target_positions"][:, :]
        )
    rms_position_error = np.mean(positions)

    # calculate RMS velocity tracking error
    velocities = []
    for i in range(len(data["agent_velocities"])):
        velocities.append(0)
        velocities[i] = root_mean_squared_error(
            data["agent_velocities"][i], data["target_velocities"]
        )
    rms_velocity_error = np.mean(velocities)

    # calculate RMS function approximation error
    function_approx = []
    for i in range(len(data["agent_nn_output_1"])):
        function_approx.append(0)
        function_approx[i] = root_mean_squared_error(
            data["agent_nn_output_1"][i][:, :],
            data["target_accelerations"][:, :],
        )
    rms_function_approx_error = np.mean(function_approx)

    # calculate RMS interaction error
    interaction_approx = []
    for i in range(len(data["agent_nn_output_2"])):
        interaction_approx.append(0)
        interaction_approx[i] = root_mean_squared_error(
            data["agent_nn_output_2"][i][:, :], data["agent_inter"][i][:, :]
        )
    rms_interaction_approx_error = np.mean(interaction_approx)

    # calculate RMS control effort
    control_effort = []
    for i in range(len(data["agent_control_effort"])):
        control_effort.append(0)
        control_effort[i] = root_mean_squared_error(
            data["agent_control_effort"][i],
            np.zeros_like(data["agent_control_effort"][i]),
        )
    rms_control_effort = np.mean(control_effort)

    params_dict = {
        "label": label,
        "trial_number": trial_number,
        "rms_position_error": rms_position_error,
        "rms_velocity_error": rms_velocity_error,
        "rms_function_approx_error": rms_function_approx_error,
        "rms_interaction_approx_error": rms_interaction_approx_error,
        "rms_control_effort": rms_control_effort,
    }

    return params_dict


if __name__ == "__main__":
    # global variables
    # specify which topologies you wish to process
    topologies = [
        "acyclic",
        "complete",
        "path",
        "ring",
        "star",
    ]  # "acyclic", "complete", "path", "ring", "star"
    architectures = ["DNN", "GNN"]
    metrics = [
        "rms_position_error",
        "rms_velocity_error",
        "rms_function_approx_error",
        "rms_interaction_approx_error",
        "rms_control_effort",
    ]
    num_agents = 6
    tf = 30
    dt = 0.005

    # storage struct
    # {'topology': {'NN': {'metric': (mean,std)}}}

    stat_storage = {
        topology: {
            "DNN": {metric: [] for metric in metrics},
            "GNN": {metric: [] for metric in metrics},
        }
        for topology in topologies
    }
    stat_mean_std = {
        topology: {
            "DNN": {metric: (0.0, 0.0) for metric in metrics},
            "GNN": {metric: (0.0, 0.0) for metric in metrics},
        }
        for topology in topologies
    }

    for topology in topologies:
        for trial in import_topology(topology):
            # append all data to relevant lists
            trial_data = calculate_params(trial)
            for metric in trial_data.keys():
                if metric != "label" and metric != "trial_number":
                    stat_storage[topology][trial_data["label"]][metric].append(
                        trial_data[metric]
                    )

    for topology in topologies:
        for architecture in architectures:
            for metric in metrics:
                stat_mean_std[topology][architecture][metric] = (
                    float(np.mean(stat_storage[topology][architecture][metric])),
                    float(np.std(stat_storage[topology][architecture][metric])),
                )

    pd_data = []
    for topology, archs in stat_mean_std.items():
        for arch, metrics in archs.items():
            for metric, (mean, std) in metrics.items():
                pd_data.append(
                    {
                        "Topology": topology,
                        "Architecture": arch,
                        "Metric": metric,
                        "Mean": round(mean, 4),
                        "Std_Dev": round(std, 4),
                    }
                )
    df = pd.DataFrame(pd_data)

    print("\nRun Statistics:")
    print(df.to_string(index=False))

    # save paper data to csv
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, "data", "simulation_stats.csv")
    df.to_csv(csv_path, index=False)
