import copy
import csv
import os
import pickle
from typing import Literal, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.stats import levene, shapiro, ttest_ind
from sklearn.metrics import root_mean_squared_error

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 8,
        "font.size": 8,
        "legend.fontsize": 5.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}\usepackage{mathrsfs}",
    }
)


def time_vector(tf: float, dt: float) -> NDArray[np.float64]:
    """
    Helper function to create a time vector
    """
    time_steps = int(tf / dt)
    return np.linspace(0, tf, time_steps, dtype=np.float64)


def calculate_mean_rms_error(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    calc_type: Optional[Literal["init", "trans"]],
    split: int = 1000,
) -> tuple[NDArray[np.float64], float]:
    """
    Generic function to calculate mean error and mean RMS error for a single simulation

    Args: (n = dimension of signal, T = number of time steps, d = number of agents)
        X:                 Estimated signal, shape (n,T,d)
        Y:                 True signal, shape (n,T,d)
        calc_type:         "init"    for initial
                           "trans"  for transient
                           None     for entire duration
        split:             Time step to split between initial and transient

    Returns:
        mean_error_signal: Mean error signal, shape (T,)
        rms_error:         RMS error, scalar
    """

    T = X.shape[1]
    assert T == Y.shape[1], "Dimension mismatch"
    r = min(split, T)

    if calc_type == "init":
        X = X[:, :r, :]
        Y = Y[:, :r, :]
    elif calc_type == "trans":
        if T > split:
            X = X[:, r:, :]
            Y = Y[:, r:, :]
        else:
            X = np.zeros_like(X)
            Y = np.zeros_like(Y)
    elif calc_type is None:
        pass

    error_signal = X - Y  # shape (n,T,d)
    norm_signal = np.linalg.norm(error_signal, axis=0)
    mean_norm_signal = np.mean(norm_signal, axis=1)  # shape (n,T)
    # mean_error_signal = np.linalg.norm(mean_error_signal, axis=0)  # shape (T,)

    rms_error = []
    for i in range(X.shape[2]):  # iterate over agents
        rms_error.append(root_mean_squared_error(X[:, :, i], Y[:, :, i]))

    mean_rms_error = np.sum(rms_error) / X.shape[2]  # average over agents

    return mean_norm_signal, mean_rms_error


def plot_error_signal(
    dt: float,
    tf: float,
    error_signals: NDArray[np.float64],
    ylabel: str,
    xlabel: str,
    legend: list[str],
    title: Optional[str] = None,
    ylim: Optional[tuple] = None,
    xlim: Optional[tuple] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Optional[Figure], Axes]:
    """
    Generic function to plot error signals for multiple simulations.

    Args: (n = dimension of signal, T = number of time steps, s = number of simulations)
        dt:            time step
        tf:            final time
        error_signals: shape (n,T,s)
        title:         plot title
        ylabel:        y-axis label
        xlabel:        x-axis label
        ylim:          y-axis limits
        xlim:          x-axis limits
        legend:        list of legend labels, length s
        ax:            matplotlib axes, if None, a new figure and axes will be created

    Returns:
        created_fig:   (optional) matplotlib figure
        ax:            matplotlib axes
    """

    # matplotlib.use("TkAgg")

    time = time_vector(tf, dt)

    created_fig = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = fig

    if title is not None:
        ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    # ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    for i in range(error_signals.shape[1]):  # iterate over simulations
        ax.plot(
            time,
            error_signals[:, i],
            label=legend[i],
            linewidth=1,
        )

    ax.legend(loc="best", fontsize=6, frameon=True, edgecolor="black")

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    return created_fig, ax


def import_data(filename):
    # get data path
    file_path = os.path.join(os.getcwd(), "data", filename)
    # open file with pickle
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def import_topology(topology: str, trial_number: Optional[int] = None):
    # import all files with given topology in data folder
    # use import_data function to import data
    # return list of data
    data_list = []
    for file in os.listdir(os.path.join(os.getcwd(), "data")):
        if file.startswith(topology):
            if trial_number is None:
                data_list.append(import_data(file))
            elif file.endswith(f"trial{trial_number}.pkl"):
                data_list.append(import_data(file))
    return data_list


def calculate_params(data, dt):

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


def access_figure(
    topology: str,
    trial_number: int,
    tf: float,
    dt: float,
    num_agents: int,
    fig_dir: str,
):

    # load list of data having the correct topology and trial number
    simulation_data_list = import_topology(topology=topology, trial_number=trial_number)

    position_tracking_error = np.zeros(
        (int(tf / dt), len(simulation_data_list))
    )  # shape (T,s)
    # velocity tracking error
    velocity_tracking_error = np.zeros_like(position_tracking_error)
    target_approx_error = np.zeros_like(position_tracking_error)
    inter_approx_error = np.zeros_like(position_tracking_error)

    for i, simulation_data in enumerate(simulation_data_list):
        agent_positions = np.zeros(
            (
                3,
                int(tf / dt),
                num_agents,
            )
        )  # shape (3,T,d)
        agent_velocities = np.zeros_like(agent_positions)
        target_positions = np.zeros_like(agent_positions)
        target_velocities = np.zeros_like(agent_positions)
        target_accelerations = np.zeros_like(agent_positions)
        interaction_dynamics = np.zeros_like(agent_positions)
        nn_1_outputs = np.zeros_like(agent_positions)
        nn_2_outputs = np.zeros_like(agent_positions)
        control_efforts = np.zeros_like(agent_positions)

        for j, agent_data in enumerate(simulation_data["agent_positions"]):
            agent_positions[:, :, j] = agent_data
        for j, agent_data in enumerate(simulation_data["agent_velocities"]):
            agent_velocities[:, :, j] = agent_data
        for j, agent_data in enumerate(simulation_data["agent_nn_output_1"]):
            nn_1_outputs[:, :, j] = agent_data
        for j, agent_data in enumerate(simulation_data["agent_nn_output_2"]):
            nn_2_outputs[:, :, j] = agent_data
        for j, agent_data in enumerate(simulation_data["agent_inter"]):
            interaction_dynamics[:, :, j] = agent_data
        for j, agent_data in enumerate(simulation_data["agent_control_effort"]):
            control_efforts[:, :, j] = agent_data
        for j in range(num_agents):
            target_positions[:, :, j] = simulation_data["target_positions"]
            target_velocities[:, :, j] = simulation_data["target_velocities"]
            target_accelerations[:, :, j] = simulation_data["target_accelerations"]

        position_tracking_error[:, i], _ = calculate_mean_rms_error(
            X=agent_positions, Y=target_positions, calc_type=None
        )
        velocity_tracking_error[:, i], _ = calculate_mean_rms_error(
            X=agent_velocities, Y=target_velocities, calc_type=None
        )
        target_approx_error[:, i], _ = calculate_mean_rms_error(
            X=target_accelerations, Y=nn_1_outputs, calc_type=None
        )
        inter_approx_error[:, i], _ = calculate_mean_rms_error(
            X=interaction_dynamics, Y=nn_2_outputs, calc_type=None
        )

    # plot in 2x2 grid, ([pos_error, vel_error], [target_error, inter_error])
    fig, axs = plt.subplots(2, 2, figsize=(6, 5))

    # for ax in axs.flat:
    #     ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # plot position tracking error
    plot_error_signal(
        dt=dt,
        tf=tf,
        error_signals=position_tracking_error,
        ylabel=r"$\frac{1}{N} \sum_{i\in V} \lVert e_i \rVert$ $[\mathrm{m}]$",
        xlabel=r"$t\ [\mathrm{s}]$",
        ylim=((0, np.max(position_tracking_error * 1.125))),
        xlim=((0, 30)),
        legend=[simulation_data["label"] for simulation_data in simulation_data_list],
        ax=axs[(0, 0)],
    )
    # plot velocity tracking error
    plot_error_signal(
        dt=dt,
        tf=tf,
        error_signals=velocity_tracking_error,
        ylabel=r"$\frac{1}{N} \sum_{i\in V} \lVert \dot{e}_i \rVert$ $[ \mathrm{m}\slash\mathrm{s} ]$",
        xlabel=r"$t\ [\mathrm{s}]$",
        ylim=((0, np.max(velocity_tracking_error * 1.125))),
        xlim=((0, 30)),
        legend=[simulation_data["label"] for simulation_data in simulation_data_list],
        ax=axs[(0, 1)],
    )
    # plot target approx error
    plot_error_signal(
        dt=dt,
        tf=tf,
        error_signals=target_approx_error,
        ylabel=r"$\frac{1}{N}\sum_{i\in V} \lVert \tilde{\phi}_{1,i} \rVert$ $[\mathrm{m}\slash\mathrm{s}^2]$",
        xlabel=r"$t\ [\mathrm{s}]$",
        ylim=((0, np.max(target_approx_error * 1.125))),
        xlim=((0, 30)),
        legend=[simulation_data["label"] for simulation_data in simulation_data_list],
        ax=axs[(1, 0)],
    )
    # plot inter approx error
    plot_error_signal(
        dt=dt,
        tf=tf,
        error_signals=inter_approx_error,
        ylabel=r"$\frac{1}{N}\sum_{i\in V} \lVert \tilde{\phi}_{2,i} \rVert$ $[\mathrm{m}\slash\mathrm{s}^2]$",
        xlabel=r"$t\ [\mathrm{s}]$",
        ylim=((0, np.max(inter_approx_error * 1.125))),
        xlim=((0, 30)),
        legend=[simulation_data["label"] for simulation_data in simulation_data_list],
        ax=axs[(1, 1)],
    )

    plt.tight_layout()

    current_dir = os.getcwd()
    results_dir = os.path.join(current_dir, fig_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    png_filename = os.path.join(
        results_dir,
        f"{topology}_{num_agents}agents_access_error_{tf}s.png",
    )
    pdf_filename = os.path.join(
        results_dir,
        f"{topology}_{num_agents}agents_access_error_{tf}s.pdf",
    )
    plt.savefig(png_filename, dpi=300)
    plt.savefig(pdf_filename, dpi=300)

    plt.show()


def make_figure(
    topology: str,
    trial_number: int,
    tf: float,
    dt: float,
    num_agents: int,
    fig_dir: str,
):

    # load list of data having the correct topology and trial number
    simulation_data_list = import_topology(topology=topology, trial_number=trial_number)

    position_tracking_error = np.zeros(
        (int(tf / dt), len(simulation_data_list))
    )  # shape (T,s)
    # velocity tracking error
    velocity_tracking_error = np.zeros_like(position_tracking_error)

    for i, simulation_data in enumerate(simulation_data_list):
        agent_positions = np.zeros(
            (
                3,
                int(tf / dt),
                num_agents,
            )
        )  # shape (3,T,d)
        agent_velocities = np.zeros_like(agent_positions)
        target_positions = np.zeros_like(agent_positions)
        target_velocities = np.zeros_like(agent_positions)
        # target_accelerations = np.zeros_like(agent_positions)
        # interaction_dynamics = np.zeros_like(agent_positions)
        # nn_1_outputs = np.zeros_like(agent_positions)
        # nn_2_outputs = np.zeros_like(agent_positions)
        # control_efforts = np.zeros_like(agent_positions)

        for j, agent_data in enumerate(simulation_data["agent_positions"]):
            agent_positions[:, :, j] = agent_data
        for j, agent_data in enumerate(simulation_data["agent_velocities"]):
            agent_velocities[:, :, j] = agent_data
        for j in range(num_agents):
            target_positions[:, :, j] = simulation_data["target_positions"]
            target_velocities[:, :, j] = simulation_data["target_velocities"]

        position_tracking_error[:, i], _ = calculate_mean_rms_error(
            X=agent_positions, Y=target_positions, calc_type=None
        )
        velocity_tracking_error[:, i], _ = calculate_mean_rms_error(
            X=agent_velocities, Y=target_velocities, calc_type=None
        )

    # plot in 1x2 grid, pos_error and vel_error
    fig, axs = plt.subplots(1, 2, figsize=(6, 2))

    # for ax in axs.flat:
    #     ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # plot position tracking error
    plot_error_signal(
        dt=dt,
        tf=tf,
        error_signals=position_tracking_error,
        ylabel=r"$\frac{1}{N} \sum_{i\in V} \lVert e_i \rVert$ $[\mathrm{m}]$",
        xlabel=r"$t\ [\mathrm{s}]$",
        ylim=((0, np.max(position_tracking_error * 1.125))),
        xlim=((0, 30)),
        legend=[simulation_data["label"] for simulation_data in simulation_data_list],
        ax=axs[0],
    )
    # plot velocity tracking error
    plot_error_signal(
        dt=dt,
        tf=tf,
        error_signals=velocity_tracking_error,
        ylabel=r"$\frac{1}{N} \sum_{i\in V} \lVert \dot{e}_i \rVert$ $[ \mathrm{m}\slash\mathrm{s} ]$",
        xlabel=r"$t\ [\mathrm{s}]$",
        ylim=((0, np.max(velocity_tracking_error * 1.125))),
        xlim=((0, 30)),
        legend=[simulation_data["label"] for simulation_data in simulation_data_list],
        ax=axs[1],
    )

    plt.tight_layout()

    current_dir = os.getcwd()
    results_dir = os.path.join(current_dir, fig_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    png_filename = os.path.join(
        results_dir,
        f"{topology}_{num_agents}agents_pos_vel_error_{tf}s.png",
    )
    pdf_filename = os.path.join(
        results_dir,
        f"{topology}_{num_agents}agents_pos_vel_error_{tf}s.pdf",
    )
    plt.savefig(png_filename, dpi=300)
    plt.savefig(pdf_filename, dpi=300)

    plt.show()


if __name__ == "__main__":
    # global variables
    topologies = ["acyclic", "complete", "path", "ring", "star"]
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

    # data struct
    # {topology: {'DNN': {'metric': (mean,std)}, 'DNN': {}}}

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
            trial_data = calculate_params(trial, dt)
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

    print("\n--- Raw Statistics ---")
    print(df.to_string(index=False))

    # statistical study
    stat_results = []
    for topology in topologies:
        for metric in metrics:
            dnn_data = stat_storage[topology]["DNN"][metric]
            gnn_data = stat_storage[topology]["GNN"][metric]

            # sw test
            dnn_w, dnn_p = shapiro(dnn_data)
            gnn_w, gnn_p = shapiro(gnn_data)

            # levene
            lev_w, lev_p = levene(dnn_data, gnn_data)

            # one-sided t
            # pass is when lev_p > 0.05
            is_equal_check = bool(lev_p > 0.05)
            ttest_obj = ttest_ind(
                dnn_data, gnn_data, equal_var=False, alternative="greater"
            )
            t_w = ttest_obj.statistic
            t_p = ttest_obj.pvalue

            stat_results.append(
                {
                    "Topology": topology,
                    "Metric": metric,
                    "DNN_Mean": round(np.mean(dnn_data), 4),
                    "GNN_Mean": round(np.mean(gnn_data), 4),
                    "DNN_Shapiro_p_val": round(dnn_p, 4),
                    "GNN_Shapiro_p_val": round(gnn_p, 4),
                    "Levene_p_val": round(lev_p, 4),
                    "T_Stat": round(t_w, 4),
                    "T_p_val": round(t_p, 4),
                    "Is_Significant": "Yes" if t_p < 0.05 else "No",
                }
            )

    stats_df = pd.DataFrame(stat_results)

    # save paper data to csv
    df.to_csv("simulation_stats.csv", index=False)
    stats_df.to_csv("simulation_analysis.csv", index=False)

    # plot paper figure
    access_figure(
        topology="ring",
        trial_number=21,
        tf=tf,
        dt=dt,
        num_agents=num_agents,
        fig_dir="plots/",
    )
