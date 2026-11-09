import os
import pickle
from typing import Literal, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.metrics import root_mean_squared_error

from simulation import Simulation

import utils

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 14,
        "font.size": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
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
    title: str,
    ylabel: str,
    xlabel: str,
    legend: list[str],
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

    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    for i in range(error_signals.shape[1]):  # iterate over simulations
        ax.plot(
            time,
            error_signals[:, i],
            label=legend[i],
            linewidth=1,
        )

    ax.legend(loc="best", fontsize=10, frameon=True, edgecolor="black")

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    return created_fig, ax


def plot_weights(
    dt: float,
    tf: float,
    weights: NDArray,
    title: str,
    ylabel: str,
    xlabel: str,
    xlim: Optional[tuple] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Optional[Figure], Axes]:
    """
    Plots evolution of selected weights over time

    Args:
        dt (float):
        tf (float):
        weights (np.ndarray): (num_weights,time_steps) array
    """

    time = time_vector(tf, dt)

    created_fig = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = fig

    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.plot(time, weights.T)

    return created_fig, ax


def plot_3d_path(
    path_signal: NDArray[np.float64],
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    ax: Optional[Axes] = None,
) -> Tuple[Optional[Figure], Axes]:
    """
    Generic function to plot 3D path for a single simulation.

    Args: (T = number of time steps, d = number of agents)
        path_signal: shape (3,T,d)
        title:       plot title
        xlabel:      x-axis label
        ylabel:      y-axis label
        zlabel:      z-axis label
        ax:          matplotlib axes, if None, a new figure and axes will be created

    Returns:
        created_fig: (optional) matplotlib figure
        ax:          matplotlib axes
    """

    created_fig = None
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        created_fig = fig

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    for i in range(path_signal.shape[2]):  # iterate over agents
        ax.plot(
            path_signal[0, :, i],
            path_signal[1, :, i],
            path_signal[2, :, i],
            label=f"Agent {i}",
        )

    ax.legend(loc="best", fontsize=14, frameon=True, edgecolor="black")

    return created_fig, ax


def pickle_results(simulation_list, data_dir):
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


def display_results(simulations: list[Simulation], fig_dir: str, show_plot: bool) -> None:
    """
    Creates and saves plots and error metrics for a list of Simulation objects

    Args:
        simulations (list[Simulation]): List of completed Simulation objects
        fig_dir (str): Desired directory to save plots and text files containing results
        show_plot (bool): Displays plots if True
    
    Returns:
        None: Saves figures
    """
    # position tracking error
    position_tracking_error = np.zeros(
        (int(simulations[0].config.tf / simulations[0].config.dt), len(simulations))
    )  # shape (T,s)
    # velocity tracking error
    velocity_tracking_error = np.zeros_like(position_tracking_error)

    # target dynamics approximation error
    target_approximation_error = np.zeros_like(position_tracking_error)
    # target dynamics
    global_target_dynamics = np.zeros_like(position_tracking_error)
    # nn_1_outputs
    global_nn_1_output = np.zeros_like(position_tracking_error)
    
    # interaction approx error
    interaction_dynamics_error = np.zeros_like(position_tracking_error)
    # interaction dynamics
    global_interaction_dynamics = np.zeros_like(position_tracking_error)
    # nn_2 outputs
    global_nn_2_output = np.zeros_like(position_tracking_error)

    # control effort
    control_effort = np.zeros_like(position_tracking_error)

    # store RMS for text output
    position_tracking_error_rms_vals = []
    velocity_tracking_error_rms_vals = []
    target_approximation_error_rms_vals = []
    interaction_approximation_error_rms_vals = []
    control_effort_rms_vals = []
    for i, simulation in enumerate(simulations):
        agent_positions = np.zeros(
            (
                3,
                int(simulation.config.tf / simulation.config.dt),
                simulation.config.num_agents,
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
        for j, agent in enumerate(simulation.agents):
            agent_positions[:, :, j] = agent.positions
            agent_velocities[:, :, j] = agent.velocities
            target_positions[:, :, j] = simulation.target.positions
            target_velocities[:, :, j] = simulation.target.velocities
            target_accelerations[:, :, j] = simulation.target.accelerations
            interaction_dynamics[:, :, j] = agent.interaction_dynamics
            nn_1_outputs[:, :, j] = agent.nn_1_output
            nn_2_outputs[:, :, j] = agent.nn_2_output
            control_efforts[:, :, j] = agent.control_input

        position_tracking_error[:, i], pte_rms = calculate_mean_rms_error(
            X=agent_positions, Y=target_positions, calc_type=None
        )
        velocity_tracking_error[:, i], vte_rms = calculate_mean_rms_error(
            X=agent_velocities, Y=target_velocities, calc_type=None
        )

        target_approximation_error[:, i], tae_rms = calculate_mean_rms_error(
            X=target_accelerations, Y=nn_1_outputs, calc_type=None
        )
        global_nn_1_output[:,i], _ =  calculate_mean_rms_error(
            X=np.zeros_like(nn_1_outputs), Y=nn_1_outputs, calc_type=None
        )
        global_target_dynamics[:,i], _ = calculate_mean_rms_error(
            X=target_accelerations, Y=np.zeros_like(nn_1_outputs), calc_type=None
        )

        interaction_dynamics_error[:, i], id_rms = calculate_mean_rms_error(
            X=interaction_dynamics, Y=nn_2_outputs, calc_type=None
        )
        global_nn_2_output[:,i], _ =  calculate_mean_rms_error(
            X=np.zeros_like(nn_1_outputs), Y=nn_2_outputs, calc_type=None
        )
        global_interaction_dynamics[:,i], _ = calculate_mean_rms_error(
            X=interaction_dynamics, Y=np.zeros_like(nn_1_outputs), calc_type=None
        )
        
        control_effort[:, i], ce_rms = calculate_mean_rms_error(
            X=control_efforts, Y=np.zeros_like(control_efforts), calc_type=None
        )

        # store RMS values
        position_tracking_error_rms_vals.append(pte_rms)
        velocity_tracking_error_rms_vals.append(vte_rms)
        target_approximation_error_rms_vals.append(tae_rms)
        interaction_approximation_error_rms_vals.append(id_rms)
        control_effort_rms_vals.append(ce_rms)

    # split into three figs per dixon suggestion
    # fig 1: control (pte, vte, control effort)
    control_fig, control_axs = plt.subplots(3,1,figsize=(8.5,10))
    for ax in control_axs.flat:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    # plot position tracking error
    plot_error_signal(
        dt=simulations[0].config.dt,
        tf=simulations[0].config.tf,
        error_signals=position_tracking_error,
        title="Position Tracking Error",
        ylabel=r"$\frac{1}{N}\sum_{i\in V} \lVert e_i \rVert$ $[\mathrm{m}]$",
        xlabel=r"$t\ [\mathrm{s}]$",
        ylim=((0, np.max(position_tracking_error * 1.125))),
        xlim=((0, simulations[0].config.tf)),
        legend=[sim.label for sim in simulations],
        ax=control_axs[0],
    )
    # plot velocity tracking error
    plot_error_signal(
        dt=simulations[0].config.dt,
        tf=simulations[0].config.tf,
        error_signals=velocity_tracking_error,
        title="Velocity Tracking Error",
        ylabel=r"$\frac{1}{N}\sum_{i\in V} \lVert \dot{e}_i \rVert$ $[ \mathrm{m}\slash\mathrm{s} ]$",
        xlabel=r"$t\ [\mathrm{s}]$",
        ylim=((0, np.max(velocity_tracking_error * 1.125))),
        xlim=((0, simulations[0].config.tf)),
        legend=[sim.label for sim in simulations],
        ax=control_axs[1],
    )
    # plot control effort
    plot_error_signal(
        dt=simulations[0].config.dt,
        tf=simulations[0].config.tf,
        error_signals=control_effort,
        title="Control Effort",
        ylabel=r"$\frac{1}{N}\sum_{i\in V} \lVert u_i \rVert$ $[\mathrm{m}\slash\mathrm{s}^2]$",
        xlabel=r"$t\ [\mathrm{s}]$",
        ylim=((0, np.max(control_effort * 1.125))),
        xlim=((0, simulations[0].config.tf)),
        legend=[sim.label for sim in simulations],
        ax=control_axs[2],
    )

    # make dir (do once)
    dirname = os.path.dirname(__file__)
    results_dir = os.path.join(dirname, fig_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # save and show
    plt.tight_layout()
    png_filename = os.path.join(
        results_dir,
        f"{simulations[0].config.topology}_{simulations[0].config.num_agents}agents_{simulations[0].config.tf}s_control.png",
    )
    pdf_filename = os.path.join(
        results_dir,
        f"{simulations[0].config.topology}_{simulations[0].config.num_agents}agents_{simulations[0].config.tf}s_control.pdf",
    )
    plt.savefig(png_filename, dpi=300)
    plt.savefig(pdf_filename, dpi=300)
    if show_plot:
        plt.show()

    # fig 2: nn_1 (fpe, DNN weights, GNN weights)
    nn_1_fig, nn_1_axs = plt.subplots(3,1,figsize=(8.5,10))
    for ax in nn_1_axs.flat:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    # plot target dynamics approximation error
    plot_error_signal(
        dt=simulations[0].config.dt,
        tf=simulations[0].config.tf,
        error_signals=target_approximation_error,
        title="Target Dynamics Approximation Error",
        ylabel=r"$\frac{1}{N}\sum_{i\in V} \lVert \tilde{\phi}_{1,i} \rVert$ $[\mathrm{m}\slash\mathrm{s}^2]$",
        xlabel=r"$t\ [\mathrm{s}]$",
        ylim=((0, np.max(target_approximation_error * 1.125))),
        xlim=((0, simulations[0].config.tf)),
        legend=[sim.label for sim in simulations],
        ax=nn_1_axs[0],
    )
    # plot nn_1_weights for sim 1
    plot_weights(
        dt=simulations[0].config.dt,
        tf=simulations[0].config.tf,
        weights=simulations[0].nn_1_agent_output_weights_trajectory,
        title=rf"{simulations[0].label} Weight Evolution",
        ylabel=r"$\hat{\theta}_1$",
        xlabel=r"$t\ [\mathrm{s}]$",
        xlim=((0, simulations[0].config.tf)),
        ax=nn_1_axs[1],
    )
    # plot nn_1_weights for sim 2
    plot_weights(
        dt=simulations[1].config.dt,
        tf=simulations[1].config.tf,
        weights=simulations[1].nn_1_agent_output_weights_trajectory,
        title=rf"{simulations[1].label} Weight Evolution",
        ylabel=r"$\hat{\theta}_1$",
        xlabel=r"$t\ [\mathrm{s}]$",
        xlim=((0, simulations[1].config.tf)),
        ax=nn_1_axs[2],
    )

    # save and show
    plt.tight_layout()
    png_filename = os.path.join(
        results_dir,
        f"{simulations[0].config.topology}_{simulations[0].config.num_agents}agents_{simulations[0].config.tf}s_nn1.png",
    )
    pdf_filename = os.path.join(
        results_dir,
        f"{simulations[0].config.topology}_{simulations[0].config.num_agents}agents_{simulations[0].config.tf}s_nn1.pdf",
    )
    plt.savefig(png_filename, dpi=300)
    plt.savefig(pdf_filename, dpi=300)
    if show_plot:
        plt.show()


    # fig 3: nn_2 (fpe, DNN weights, GNN weights)
    nn_2_fig, nn_2_axs = plt.subplots(3,1,figsize=(8.5,10))
    for ax in nn_2_axs.flat:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    # plot interaction dynamics error
    plot_error_signal(
        dt=simulations[0].config.dt,
        tf=simulations[0].config.tf,
        error_signals=interaction_dynamics_error,
        title="Interaction Dynamics Approximation Error",
        ylabel=r"$\frac{1}{N}\sum_{i\in V} \lVert \tilde{\phi}_{2,i} \rVert$ $[\mathrm{m}\slash\mathrm{s}^2]$",
        xlabel=r"$t\ [\mathrm{s}]$",
        ylim=((0, np.max(interaction_dynamics_error * 1.125))),
        xlim=((0, simulations[0].config.tf)),
        legend=[sim.label for sim in simulations],
        ax=nn_2_axs[0],
    )
    # plot nn_2_weights for sim 1
    plot_weights(
        dt=simulations[0].config.dt,
        tf=simulations[0].config.tf,
        weights=simulations[0].nn_2_agent_output_weights_trajectory,
        title=rf"{simulations[0].label} Weight Evolution",
        ylabel=r"$\hat{\theta}_2$",
        xlabel=r"$t\ [\mathrm{s}]$",
        xlim=((0, simulations[0].config.tf)),
        ax=nn_2_axs[1],
    )
    # plot nn_2_weights for sim 2
    plot_weights(
        dt=simulations[1].config.dt,
        tf=simulations[1].config.tf,
        weights=simulations[1].nn_2_agent_output_weights_trajectory,
        title=rf"{simulations[1].label} Weight Evolution",
        ylabel=r"$\hat{\theta}_2$",
        xlabel=r"$t\ [\mathrm{s}]$",
        xlim=((0, simulations[1].config.tf)),
        ax=nn_2_axs[2],
    )

    # save and show
    plt.tight_layout()
    png_filename = os.path.join(
        results_dir,
        f"{simulations[0].config.topology}_{simulations[0].config.num_agents}agents_{simulations[0].config.tf}s_nn2.png",
    )
    pdf_filename = os.path.join(
        results_dir,
        f"{simulations[0].config.topology}_{simulations[0].config.num_agents}agents_{simulations[0].config.tf}s_nn2.pdf",
    )
    plt.savefig(png_filename, dpi=300)
    plt.savefig(pdf_filename, dpi=300)
    if show_plot:
        plt.show()

    # print RMS values
    output_text = (
        "RMS Values:\n"
        f"Position Tracking Error RMS: {utils.rnd_result(input_vals=position_tracking_error_rms_vals)}\n"
        f"Velocity Tracking Error RMS: {utils.rnd_result(input_vals=velocity_tracking_error_rms_vals)}\n"
        f"Target Approximation Error RMS: {utils.rnd_result(input_vals=target_approximation_error_rms_vals)}\n"
        f"Interaction Approximation Error RMS: {utils.rnd_result(input_vals=interaction_approximation_error_rms_vals)}\n"
        f"Control Effort RMS: {utils.rnd_result(input_vals=control_effort_rms_vals)}\n"
    )
    print(output_text)

    txt_filename = os.path.join(
        results_dir,
        f"{simulations[0].config.topology}_{simulations[0].config.num_agents}agents_{simulations[0].config.tf}s.txt",
    )
    # save without rounding
    save_text = (
        "RMS Values:\n"
        f"Position Tracking Error RMS: {position_tracking_error_rms_vals}\n"
        f"Velocity Tracking Error RMS: {velocity_tracking_error_rms_vals}\n"
        f"Target Approximation Error RMS: {target_approximation_error_rms_vals}\n"
        f"Interaction Approximation Error RMS: {interaction_approximation_error_rms_vals}\n"
        f"Control Effort RMS: {control_effort_rms_vals}\n"
    )
    with open(txt_filename, "w") as text_file:
        text_file.write(save_text)
