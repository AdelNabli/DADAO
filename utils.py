import os
import scipy
import torch
import numpy as np
import networkx as nx


def create_pi_ij(i, j, n):
    """
    Create the tensor $(e_i - e_j) \otimes (e_i - e_j)$.
    """

    pi_ij = torch.zeros((n, n)).double()
    pi_ij[i, j] = -1
    pi_ij[j, i] = -1
    pi_ij[i, i] = 1
    pi_ij[j, j] = 1

    return pi_ij


def compute_graph_resistance(L, G):
    """
    Compute the graph's resistance using the Laplacian L.
    The resistance is defined as 
    $\max_{(i,j) \in E} \frac{1}{2} (e_i - e_j)^\top L^+ (e_i - e_j)
    
    Parameters:
        - L (np.array): A Laplacian matrix of G.
        - G (nx.graph): A NetworkX graph.
    Returns:
        - R_max (float): the worst case resistance between two edges.
    """

    n = len(L)
    # compute the pseudo inverse of L
    L_inv = scipy.linalg.pinv(L)
    R_max = 0
    e_blank = np.zeros(n)
    # Compute the resistance of each edgge
    for (i, j) in G.edges:
        e_ij = e_blank.copy()
        e_ij[i] = 1
        e_ij[j] = -1
        R_ij = 0.5 * e_ij.T @ L_inv @ e_ij
        # save the worst case resistance
        if R_ij > R_max:
            R_max = R_ij

    return R_max


def compute_resistance_edge(L, edge):
    """
    Compute the resistance of a given edge using the Laplacian L.
    This resistance is the one used in the Continuized framework,
    where there is no 1/2 factor.
    """

    n = len(L)
    # compute the
    L_inv = scipy.linalg.pinv(L)
    e_ij = np.zeros(n)
    i, j = edge
    e_ij[i] = 1
    e_ij[j] = -1
    R_ij = e_ij.T @ L_inv @ e_ij

    return R_ij


def compute_spectral_quantities(G):
    """
    Compute the different Laplacians and spectral quantities used
    for the different methods.
    
    Parameters:
        - G (nx.graph): the graph considered.
    Returns:
        - W (np.array): normalized Laplacian used in ADOM+.
        - L_normm (np.array): Laplacian of the graph weighted with
                              the uniform proba distribution on the edges,
                              used in the continuized framework.
        - chi_1_normm (float): inverse of the smallest positive eigenvalue of L_norm.
        - chi_2_norm (float): the graph's resistance computed using L_norm.
        - chi_kov (float): the parameter chi in ADOM+.                    
    """

    # Compute Laplacian of the graph
    L = nx.laplacian_matrix(G).toarray()
    # if all edges are sampled uniformly, L_norm is the Laplacian of the
    # graph weighted with edge probabilities as defined page 30 in https://arxiv.org/pdf/2106.07644.pdf
    n_edges = len(G.edges)
    L_norm = L / (n_edges)
    # smallest positive eigenvalue of L_norm
    chi_1_norm = 1 / scipy.linalg.eigh(L_norm)[0][1]
    chi_2_norm = compute_graph_resistance(L_norm, G)
    # Compute the quantities used by Kovalev et al,
    # described in page 4 of https://arxiv.org/pdf/2106.04469.pdf
    lambda_max = scipy.linalg.eigh(L)[0][-1]
    lambda_min = scipy.linalg.eigh(L)[0][1]
    chi_kov = lambda_max / lambda_min
    W = L / lambda_max

    return W, L_norm, chi_1_norm, chi_2_norm, chi_kov


def dual_grad_linear_regression(data, labels, y_i, node_i):
    """
    The squared loss for linear regression, is the following function for each worker i:
    
    f_i(\theta) = (1/m)*sum_{j=1}^m (< data_{ij},  \theta > - labels_{ij})^2
    
    Parameters:
        - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim]
                                   the data points at each worker.
        - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1]
                                 the labels of each data point.
    Returns:
        grad_star (torch.tensor): of shape [n_workers, dim],
                                  the gradient of the Fenchel conjugate of f_i
                                  evaluated at y_i.
    """
    D_i = data[node_i]
    label_i = labels[node_i].squeeze()
    n_i = len(D_i)
    grad_star = torch.linalg.pinv(D_i.T @ D_i) @ ((n_i / 2) * y_i + D_i.T @ label_i)

    return grad_star


def compute_average_distance_to_opt(X, x_star):
    """
    Compute the average : $ \frac{1}{n} \sum_{i=1}^n \Vert x_i - x^* \Vert^2 $
    
    Parameters:
        - X (torch.tensor): tensor of shape [n_workers, dim]
        - x_star (numpy array): of shape [dim]
    Returns:
        - average_error (float): the average distance over the workers to x^*
    """
    with torch.no_grad():
        # compute x_i - x_star for all i
        diff = X - torch.tensor(x_star).double()
        squared_diff = diff ** 2
        squared_norm = torch.sum(squared_diff, dim=1)
        average_error = torch.mean(squared_norm)

        return average_error.numpy()


def compute_regularity_constants(data, classification):
    """
    Given the data, compute the strong convexity and smoothness parameters for the
    Linear Regression and Logistic Regression tasks.
    
    Parameters:
        - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim],
                               contains the data points.
        - classification (bool): whether or not the task is the Logistic Regression one.
    Returns:
        - mu (float): the strong convexity parameter.
        - L (float): the smoothness parameter.
    """

    # stores the constant for each local functions
    L_i = []
    mu_i = []
    n_workers = data.shape[0]
    # If Linear Regression
    if not classification:
        for i in range(n_workers):
            D_i = data[i].double()  # of shape
            n_i = len(D_i)
            # \lambda_max
            L_i.append((2 / n_i) * torch.linalg.eigh(D_i.T @ D_i)[0][-1])
            # \lambda_min
            mu_i.append((2 / n_i) * torch.linalg.eigh(D_i.T @ D_i)[0][0])

        # take the worst case as the global constants
        mu = min(mu_i)
        L = max(L_i)
    # If l2 regularized logistic regression
    else:
        for i in range(n_workers):
            D_i = data[i]
            L_i.append(torch.mean(torch.sum(D_i ** 2, dim=1), dim=0) / 4)
        mu = 1  # force the mu param of the l2 regularization to be 1
        L_i.append(mu)
        L = max(L_i)
    return mu, L


def save_data(
    loss_list,
    time_now,
    optimizer_name,
    use_MC,
    sotchastic,
    graph_type,
    n_workers,
    is_loss_comp,
):
    """
    Saves the training loss in a unique .npy file in the 'runs' directory.
    
    Parameters:
        - loss_list (list of floats): evolution of the distance to opt during training.
        - time_now (str): the date and time of the run.
        - optimizer_name (str): name of the method used.
        - use_MC (bool): whether or not Multi-Consensus was used.
        - stochastic (bool): whether or not SGD was used.
        - graph_type (str): type of graph on which the method was run.
        - n_workers (int): the number of workers for the experiment.
        - is_loss_comp (bool): whether or not the loss was counted each gradient step,
                               or rather each communication step.
    """

    # Create the file name
    # Gradients or communication
    loss_type = "grad" * is_loss_comp + "comm" * (1 - is_loss_comp)
    # create a file name
    # If ADOM+, specifies whether or not used Multi-consensus
    if optimizer_name == "ADOMplus" and use_MC:
        file_name = "_".join(
            [optimizer_name, "MC", loss_type, graph_type, str(n_workers), time_now]
        )
    # If we run the SGD version of our method
    elif optimizer_name == "Asynchronous" and stochastic:
        file_name = "_".join(
            [optimizer_name, "SGD", loss_type, graph_type, str(n_workers), time_now]
        )
    else:
        file_name = "_".join(
            [optimizer_name, loss_type, graph_type, str(n_workers), time_now]
        )

    # Create a 'runs' directory if not existing
    root = os.getcwd()
    path_runs = os.path.join(root, "runs")
    if not os.path.isdir(path_runs):
        os.mkdir(path_runs)

    # Save the loss_list in a .npy file in the runs directory
    file_path = os.path.join(path_runs, file_name)
    np.save(file_path, np.array(loss_list))
