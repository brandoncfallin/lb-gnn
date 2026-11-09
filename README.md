![NCR Logo](./misc/banner.png)

# Lyapunov-Based Graph Neural Networks for Adaptive Control of Multi-Agent Systems

by Brandon C. Fallin, Cristian F. Nino, Omkar Sudhir Patil, Zachary I. Bell, and Warren E. Dixon

This work was submitted for consideration as a full paper in IEEE Access and is under review.

## Abstract

> Graph neural networks (GNNs) use a message passing framework in which vector messages are exchanged between graph nodes and updated using feedforward layers. The inclusion of message passing in the GNN architecture makes them ideally suited for distributed control tasks. Existing results develop GNN-based controllers to compensate for modeling uncertainties in the system dynamics. However, these results use GNNs that are pre-trained offline. This paper provides the first result on GNNs with analytic update laws derived from a Lyapunov-based analysis which are used to adaptively estimate unknown inter-agent interaction dynamics and address the second-order target tracking problem. A Lyapunov-based stability analysis is provided to guarantee exponential convergence of the target state and agent states to a neighborhood of the target state. Numerical simulations validate the theoretical findings and demonstrate improvements in position tracking error and function approximation performance by the GNN over a baseline deep neural network (DNN) architecture.

## Usage

Set up environment

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

First, configure variables in `scripts/single_run.py`. The default communication topology is `ring`. Then, run the numerical experiment. To perform a single run and plot results, use:

```
python scripts/single_run.py
python scripts/plot_results.py
```

Each run's data is saved to `/data`, and `scripts/plot_results.py` saves a `.png` and `.pdf` plot to `/figures`. 

To perform multiple runs, specify the topology in `scripts/run_averages.py` and ensure `save_results=True` in `scripts/single_run.py`. Then, run:

```
python scripts/run_averages.py
python scripts/process_data.py
```

The script `scripts/process_data.py` saves a `.csv` file with information from each run to `/data`