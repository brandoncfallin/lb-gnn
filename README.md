![NCR Logo](./misc/banner.png)

# Lyapunov-Based Graph Neural Networks for Adaptive Control of Multi-Agent Systems

by Brandon C. Fallin, Cristian F. Nino, Omkar Sudhir Patil, Zachary I. Bell, and Warren E. Dixon

This work was submitted for consideration as a full paper in IEEE Access and is under review.

## Abstract

> Graph neural networks (GNNs) have a message-passing framework in which vector messages are exchanged between graph nodes and updated using feedforward layers. The inclusion of distributed message-passing in the GNN architecture makes them ideally suited for distributed control and coordination tasks. Existing results develop GNN-based controllers to address a variety of multi-agent control problems while compensating for modeling uncertainties in the systems. However, these results use GNNs that are pre-trained offline. This paper provides the first result on GNNs with stability-driven online weight updates to address the multi-agent target tracking problem. Specifically, new Lyapunov-based distributed GNN and graph attention network (GAT)-based controllers are developed to adaptively estimate unknown target dynamics and address the second-order target tracking problem. A Lyapunov-based stability analysis is provided to guarantee exponential convergence of the target state estimates and agent states to a neighborhood of the target state. Numerical simulations show a 20.8% and 48.1% position tracking error performance improvement by the GNN and GAT architectures over a baseline DNN architecture, respectively.
