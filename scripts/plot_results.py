import os
import pickle
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
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

    time = time_vector(tf, dt)

    created_fig = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = fig

    if title is not None:
        ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)

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
    """
    Load pickle file located at the specified path

    Args:
        filename (str): File path for a single run's data

    Returns:
        dict: Data dictionary from the given run
    """

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

    # import all files with given topology in data folder
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


def plot_average_errors(
    topology: str,
    trial_number: int,
    tf: float,
    dt: float,
    num_agents: int,
    fig_dir: str,
) -> None:
    """
    Plots figure in Simulated Results section

    Args:
        topology (str): Communication topology ("ring", "star", "acyclic", "complete", "path")
        trial_number (int): Trial number for which data will be imported
        tf (float): Final time, must be <= final time in imported run
        dt (float): Time step delta, must match that of imported run
        num_agents (int): Number of agents in simulation
        fig_dir (str): Directory where figure will be saved as .png and .pdf
    Returns:
        None
    """

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
    # plot interaction approx error
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
        f"{topology}_{num_agents}agents_average_error_{tf}s.png",
    )
    pdf_filename = os.path.join(
        results_dir,
        f"{topology}_{num_agents}agents_average_error_{tf}s.pdf",
    )
    plt.savefig(png_filename, dpi=300)
    plt.savefig(pdf_filename, dpi=300)


if __name__ == "__main__":
    # global variables
    num_agents = 6
    tf = 30
    dt = 0.005

    # plot paper figure
    plot_average_errors(
        topology="ring",
        trial_number=0,
        tf=tf,
        dt=dt,
        num_agents=num_agents,
        fig_dir="figures/",
    )

    print("Results plotted for {topology} topology trial {trial_number}.")
