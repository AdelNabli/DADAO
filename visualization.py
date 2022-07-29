import torch
import scipy
import numpy as np
import networkx as nx
import pandas as pd
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
    f = plt.figure(figsize=(7, 7))
    ax0 = f.add_subplot(221)
    ax1 = f.add_subplot(222)
    ax2 = f.add_subplot(223)
    ax3 = f.add_subplot(224)
    ax0.scatter(data[0][:, 0], data[0][:, 1], c=labels[0])
    ax0.set_title("Data for worker 1")
    ax1.scatter(data[1][:, 0], data[1][:, 1], c=labels[1])
    ax1.set_title("Data for worker 2")
    ax2.scatter(data[2][:, 0], data[2][:, 1], c=labels[2])
    ax2.set_title("Data for worker 3")
    ax3.scatter(data_all[:, 0], data_all[:, 1], c=labels_all)
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
    c_all = ["#1f77b4"] * m + ["#ff7f0e"] * m + ["#d62728"] * m
    # plot the datasets for the 3 first workers
    f = plt.figure(figsize=(7, 7))
    ax0 = f.add_subplot(221)
    ax1 = f.add_subplot(222)
    ax2 = f.add_subplot(223)
    ax3 = f.add_subplot(224)
    ax0.scatter(data[0][:, 0], labels[0], c=["#1f77b4"] * len(labels[0]))
    ax0.set_title("Data for worker 1")
    ax1.scatter(data[1][:, 0], labels[1], c=["#ff7f0e"] * len(labels[0]))
    ax1.set_title("Data for worker 2")
    ax2.scatter(data[2][:, 0], labels[2], c=["#d62728"] * len(labels[0]))
    ax2.set_title("Data for worker 3")
    ax3.scatter(data_all, labels_all, c=c_all)
    ax3.set_title("Aggregated data")
    plt.show()


def plot_all_fixed_graphs(n_workers):
    """
    Plot the 6 graphs with fixed topology we considered.
    """
    list_types = ["star", "complete", "cycle", "path", "2D_grid", "barbell"]
    f = plt.figure(figsize=(9, 6))
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
        nx.draw_networkx(
            G, with_labels=False, node_color="salmon", node_size=200, ax=subplots[k]
        )
        subplots[k].set_title(graph_type, color="salmon")
    plt.tight_layout()
    plt.show()


def plot_2_time_varying_graph_sequence(n_workers):
    """
    Plot 2 sequences of 3 random geometric graphs (on which we forced the connectedness),
    with 2 different values of \chi.
    """
    f = plt.figure(figsize=(9, 6))
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
        list_G, _, _, _, _, chi = create_K_graphs(n_workers, "random_geom", radius, K=3)
        for k in range(3):
            G = list_G[k]
            nx.draw_networkx(
                G,
                with_labels=False,
                node_color="salmon",
                node_size=200,
                ax=subplots[count],
            )
            subplots[count].set_title("t = %d  (chi = %.2f)" % (k, chi), color="salmon")
            count += 1
    plt.tight_layout()
    plt.show()


def plot_losses(grad_loss, com_loss, optimizer_name, graph_type, n_workers):
    """
    Plot the evolution of the losses with respect to the total
    number of local gradient and communication steps performed.
    """
    f = plt.figure(figsize=(9, 4))
    ax0 = f.add_subplot(121)
    ax1 = f.add_subplot(122)
    ax0.plot(np.arange(len(grad_loss)), np.array(grad_loss))
    ax0.set(
        xlabel="Total number of gradient steps",
        ylabel=r"$\frac{1}{n} \sum_{i=1}^{n}\Vert x_i - x^*\Vert^2$",
        title="Gradient steps, %s graph, n = %d" % (graph_type, n_workers),
        yscale="log",
    )
    ax0.legend((optimizer_name,), loc="upper right")
    ax1.plot(np.arange(len(com_loss)), np.array(com_loss))
    ax1.set(
        xlabel="Total number of communication steps",
        ylabel=r"$\frac{1}{n} \sum_{i=1}^{n}\Vert x_i - x^*\Vert^2$",
        title="Communication steps, %s graph, n = %d" % (graph_type, n_workers),
        yscale="log",
    )
    ax1.legend((optimizer_name,), loc="upper right")
    plt.tight_layout()
    plt.show()


def print_communication_rates_fixed_topology(
    list_G, list_L_norm, chi_1_star, chi_2_star, chi_star, graph_type
):
    """
    Print the different communication rates in the fixed topology case.
    """
    G = list_G[0]
    n_edges = len(G.edges)
    n_workers = len(G)
    # rate for ADOM+
    n_rounds = int(np.ceil(chi_star * np.log(2)))
    rate_adom = n_rounds * n_edges
    # rate for DADAO and Continuized,
    # the 2 asynchronous methods
    rate_async = int(np.sqrt(2 * chi_1_star * chi_2_star))
    # rate for MSDA, using the same laplacian as in the
    # asynchronous methods
    W = list_L_norm[0]
    lamb_max = scipy.linalg.eigh(W)[0][-1]
    lamb_min = scipy.linalg.eigh(W)[0][1]
    gamma = lamb_min / lamb_max
    rate_msda = n_edges * int(1 / np.sqrt(gamma))
    print(
        " Communication rates for the different methods on the %s graph with %d nodes. \n We print the expected number of edges activated between two expected global rounds of gradient steps. \n"
        % (graph_type, n_workers)
    )
    print(
        " ADOM+ : %d \n ADOM+ with Multi-Consensus : %d \n Continuized : %d \n MSDA : %d \n DADAO : %d"
        % (n_edges, rate_adom, rate_async, rate_msda, rate_async)
    )


class Args_for_main:
    """
    Helper class to pass arguments to the main function.
    """

    def __init__(
        self,
        optimizer_name,
        data,
        labels,
        classification,
        mu,
        L,
        dim,
        n_workers,
        chi,
        chi_1,
        chi_2,
        lamb_grad,
        lamb_mix,
        t_max,
        list_G,
        list_W,
        graph_type,
        use_multi_consensus=False,
        stochastic=False,
        batch_size=1,
    ):
        self.optimizer_name = optimizer_name
        self.data = data
        self.labels = labels
        self.classification = classification
        self.mu = mu
        self.L = L
        self.dim = dim
        self.n_workers = n_workers
        self.chi = chi
        self.chi_1 = chi_1
        self.chi_2 = chi_2
        self.lamb_grad = lamb_grad
        self.lamb_mix = lamb_mix
        self.t_max = t_max
        self.list_G = list_G
        self.list_W = list_W
        self.graph_type = graph_type
        self.use_multi_consensus = use_multi_consensus
        self.stochastic = stochastic
        self.batch_size = batch_size


def plot_all_losses(
    n_workers,
    grad_loss_dadao=[],
    com_loss_dadao=[],
    grad_loss_adom=[],
    com_loss_adom=[],
    grad_loss_adom_mc=[],
    com_loss_adom_mc=[],
    grad_loss_msda=[],
    com_loss_msda=[],
    grad_loss_continuized=[],
    com_loss_continuized=[],
    graph_type=None,
    id_max=None,
    compression=0.0,
):
    """
    Plot the evolution of the distance to the true optimal value with the number of gradient and communication
    steps for all methods.
    
    Parameters:
        - n_workers (int): number of workers considered.
        - x_loss_x (list): evolution of the distance to the true optimal value during training.
        - graph_type (str): the type of graph considered.
        - id_max (int): the maximum index considered for the plots.
        - compression (float): in [0, 100], percentage of data to discard for the plots,
                               in order to speed up the plotting process.
    """
    loss_list_grad = [
        grad_loss_adom_mc,
        grad_loss_adom,
        grad_loss_continuized,
        grad_loss_msda,
        grad_loss_dadao,
    ]
    loss_list_com = [
        com_loss_adom_mc,
        com_loss_adom,
        com_loss_continuized,
        com_loss_msda,
        com_loss_dadao,
    ]
    loss_lists = [loss_list_grad, loss_list_com]
    is_grad = [True, False]
    f = plt.figure(figsize=(9, 4))
    ax0 = f.add_subplot(121)
    ax1 = f.add_subplot(122)
    axes = [ax0, ax1]
    for i in range(2):
        loss_list_i = loss_lists[i]
        # first, cap the lists if id_max is not None
        loss_list_bis = []
        for loss_list in loss_list_i:
            if id_max is not None and len(loss_list) > id_max:
                loss_list_bis.append(loss_list[: int(id_max)])
            else:
                loss_list_bis.append(loss_list)
        # then, gather the max length of the lists
        max_len = max([len(list_loss) for list_loss in loss_list_bis])
        # create a data dictionnary
        data_dict = dict()
        # initialize the data
        data_dict["Methods"] = []
        data_dict["x_axis"] = []
        data_dict["loss"] = []
        for k, loss_list in enumerate(loss_list_bis):
            len_list = len(loss_list)
            if len_list > 0:
                # fill missing values with np.nan
                array_loss = list(np.array(loss_list)) + [np.nan] * (max_len - len_list)
                # compress
                step_size = int(1 / (1 - compression / 100))
                compressed_len = int(len(array_loss) / step_size)
                new_array = []
                x_axis = []
                # keep only a fraction of the data if the compression factor is not 0
                for j in range(compressed_len):
                    new_array.append(array_loss[j * step_size])
                    x_axis.append(j * step_size)
                array_loss = new_array
                # fill the data dict
                data_dict["x_axis"] = data_dict["x_axis"] + x_axis
                data_dict["loss"] = data_dict["loss"] + array_loss
                if k == 0:
                    data_dict["Methods"] = data_dict["Methods"] + ["ADOM+ M.-C."] * len(
                        x_axis
                    )
                elif k == 1:
                    data_dict["Methods"] = data_dict["Methods"] + ["ADOM+"] * len(
                        x_axis
                    )
                elif k == 2:
                    data_dict["Methods"] = data_dict["Methods"] + ["Continuized"] * len(
                        x_axis
                    )
                elif k == 3:
                    data_dict["Methods"] = data_dict["Methods"] + ["MSDA"] * len(x_axis)
                elif k == 4:
                    data_dict["Methods"] = data_dict["Methods"] + ["DADAO"] * len(
                        x_axis
                    )

        # PLot the graphs with seaborn
        df = pd.DataFrame(data_dict)
        sns.set(font_scale=1.05, style="white")
        sns.lineplot(
            data=df,
            x="x_axis",
            y="loss",
            style="Methods",
            hue="Methods",
            palette="magma",
            ax=axes[i],
        )
        legend = (
            "Total number of gradients" * is_grad[i]
            + (1 - is_grad[i]) * "Total number of communications"
        )
        title = (
            "Grad. steps" * is_grad[i]
            + (1 - is_grad[i]) * "Com. steps"
            + ", %s graph, n = %d" % (graph_type, n_workers)
        )
        axes[i].set(
            xlabel=legend,
            ylabel=r"$\frac{1}{n} \sum_{i=1}^{n}\Vert x_i - x^*\Vert^2$",
            title=title,
            yscale="log",
            ylim=(1e-5, 1e5),
        )
        axes[i].legend(loc="center right")
    plt.tight_layout()
    plt.show()
