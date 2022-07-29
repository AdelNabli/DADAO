import torch
import scipy
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from data import generate_one_connected_graph, create_K_graphs


def plot_3_classif(data, labels):
    """
    Scatter 2D data points for the 3 first workers
    for the Binary Classification task.
    """
    # aggregate the data
    data_all = torch.cat([data[k] for k in range(3)], dim=0)
    labels_all = torch.cat([labels[k] for k in range(3)], dim=0)
    # plot the datasets for the 3 first workers
    f = plt.figure(figsize=(7,7))
    ax0 = f.add_subplot(221)
    ax1 = f.add_subplot(222)
    ax2 = f.add_subplot(223)
    ax3 = f.add_subplot(224)
    ax0.scatter(data[0][:,0], data[0][:,1], c=labels[0])
    ax0.set_title("Data for worker 1")
    ax1.scatter(data[1][:,0], data[1][:,1], c=labels[1])
    ax1.set_title("Data for worker 2")
    ax2.scatter(data[2][:,0], data[2][:,1], c=labels[2])
    ax2.set_title("Data for worker 3")
    ax3.scatter(data_all[:,0], data_all[:,1], c=labels_all)
    ax3.set_title("Aggregated data")
    plt.show()


def plot_3_reg(data, labels):
    """
    Plot the 1D data points and labels for the 3 first workers
    for the Linear regression task.
    """
    # aggregate the data
    data_all = torch.cat([data[k] for k in range(3)], dim=0)
    labels_all = torch.cat([labels[k] for k in range(3)], dim=0)
    m = len(labels[0])
    c_all = ['#1f77b4']*m + ['#ff7f0e']*m + ['#d62728']*m
    # plot the datasets for the 3 first workers
    f = plt.figure(figsize=(7,7))
    ax0 = f.add_subplot(221)
    ax1 = f.add_subplot(222)
    ax2 = f.add_subplot(223)
    ax3 = f.add_subplot(224)
    ax0.scatter(data[0][:,0], labels[0], c=['#1f77b4']*len(labels[0]))
    ax0.set_title("Data for worker 1")
    ax1.scatter(data[1][:,0], labels[1], c=['#ff7f0e']*len(labels[0]))
    ax1.set_title("Data for worker 2")
    ax2.scatter(data[2][:,0], labels[2], c=['#d62728']*len(labels[0]))
    ax2.set_title("Data for worker 3")
    ax3.scatter(data_all, labels_all, c=c_all)
    ax3.set_title("Aggregated data")
    plt.show()


def plot_all_fixed_graphs(n_workers):
    """
    Plot the 6 graphs with fixed topology we considered.
    """
    list_types = ['star', 'complete', 'cycle', 'path', '2D_grid', 'barbell']
    f = plt.figure(figsize = (9, 6))
    ax0 = f.add_subplot(231)
    ax1 = f.add_subplot(232)
    ax2 = f.add_subplot(233)
    ax3 = f.add_subplot(234)
    ax4 = f.add_subplot(235)
    ax5 = f.add_subplot(236)
    subplots = [ax0, ax1, ax2, ax3, ax4, ax5]
    for k in range(6):
        graph_type = list_types[k]
        G = generate_one_connected_graph(n_workers, graph_type, radius=None)
        nx.draw_networkx(G, with_labels=False, node_color='salmon', node_size=200, ax=subplots[k])
        subplots[k].set_title(graph_type, color='salmon')
    plt.tight_layout()
    plt.show()


def plot_2_time_varying_graph_sequence(n_workers):
    """
    Plot 2 sequences of 3 random geometric graphs (on which we forced the connectedness),
    with 2 different values of \chi.
    """
    f = plt.figure(figsize = (9, 6))
    ax0 = f.add_subplot(231)
    ax1 = f.add_subplot(232)
    ax2 = f.add_subplot(233)
    ax3 = f.add_subplot(234)
    ax4 = f.add_subplot(235)
    ax5 = f.add_subplot(236)
    subplots = [ax0, ax1, ax2, ax3, ax4, ax5]
    list_radius = [0.25, 0.6]
    count = 0
    for radius in list_radius:
        list_G, _, _, _, _, chi = create_K_graphs(n_workers, 'random_geom', radius, K=3)
        for k in range(3):
            G = list_G[k]
            nx.draw_networkx(G, with_labels=False, node_color='salmon', node_size=200, ax=subplots[count])
            subplots[count].set_title('t = %d  (chi = %.2f)'%(k,chi), color='salmon')
            count += 1
    plt.tight_layout()
    plt.show()


def print_communication_rates_fixed_topology(list_G, list_L_norm, chi_1_star, chi_2_star, chi_star, graph_type):
    """
    Print the different communication rates in the fixed topology case.
    """
    G = list_G[0]
    n_edges = len(G.edges)
    n_workers = len(G)
    # rate for ADOM+
    n_rounds = int(np.ceil(chi_star * np.log(2)))
    rate_adom = n_rounds*n_edges
    # rate for DADAO and Continuized,
    # the 2 asynchronous methods
    rate_async = int(np.sqrt(2*chi_1_star*chi_2_star))
    # rate for MSDA, using the same laplacian as in the
    # asynchronous methods
    W = list_L_norm[0]
    lamb_max = scipy.linalg.eigh(W)[0][-1]
    lamb_min = scipy.linalg.eigh(W)[0][1]
    gamma = lamb_min/lamb_max
    rate_msda = n_edges*int(1/np.sqrt(gamma))
    print(" Communication rates for the different methods on the %s graph with %d nodes. \n We print the expected number of edges activated between two expected global rounds of gradient steps. \n"%(graph_type, n_workers))
    print(' ADOM+ : %d \n ADOM+ with Multi-Consensus : %d \n Continuized : %d \n MSDA : %d \n DADAO : %d'%(n_edges, rate_adom, rate_async, rate_msda, rate_async))
        