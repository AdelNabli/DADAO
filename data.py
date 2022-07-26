import torch
import numpy as np
import scipy
import networkx as nx
from sklearn import datasets
from utils import compute_spectral_quantities


def create_data(n_samples, dim, n_workers, classification):
    """
    Create a collection of n_workers different random datasets,
    either for the task of binary classification or linear regression.
    
    Parameters:
        - n_samples (int): the total number of data points.
        - dim (int): the dimension of each data point.
        - n_workers (int): the number of workers.
        - classification (bool): whether we want data for 
                                binary classification or linear regression.
    Returns:
        - X_workers (torch.tensor): tensor of shape [n_workers, n_samples//n_workers, dim]
                                    containing all the data points from the n_workers.
        - y_workers (torch.tensor): tensor of shape [n_workers, n_samples//n_workers, 1]
                                    containing all the targets values.
    """

    X_all, y_all = [], []

    # Create a random dataset (generated by different parameters) for each worker
    for k in range(n_workers):
        # compute the n_data_per_worker
        n_data = n_samples // n_workers
        if classification:
            # create a random dataset for the task of binary classification
            X, y = datasets.make_classification(n_data, dim, n_classes=2, n_redundant=0)
            # put the labels in {-1, 1}
            y = y * 2 - 1
        else:
            # create a random dataset for the task of linear regression
            X, y = datasets.make_regression(n_data, dim, coef=False, noise=10)
            # introduce additional variance in the slopes produced
            coef_slope = np.random.uniform(0.01, 5)
            y = coef_slope * y
        # convert to torch
        X, y = torch.from_numpy(X).double(), torch.from_numpy(y).double()
        X_all.append(X.unsqueeze(0))
        y_all.append(y.unsqueeze(0))
    # Concatenate all the n_workers datasets in a tensor
    X_workers = torch.cat(X_all)
    y_workers = torch.cat(y_all).unsqueeze(-1)

    return X_workers, y_workers


def create_asynchronous_jump_processes(t_max, lamb_grad, lamb_mix):
    """
    Given the parameters lamb_grad and lamb_mix of each of the
    two different poisson processes, returns a list giving the
    nature of next jumps (gradient or consensus) and the list
    of waiting times between two consecutive gradient steps.
    
    Parameters:
        - t_max (float): the time of the whole process.
        - lamb_grad (float): the average of "how many jumps"
                             happen in 1 unit of time for the gradient process.
        - lamb_mix (float): the average of "how many jumps"
                            happen in 1 unit of time for the consensus process.
    Returns:
        - take_grad_step (list of bool): list all the jumps from the 2 processes
                                         in appearing order and gives whether
                                         or not it is a gradient step.
        - t_event (list of floats): the ordered list of the times an event took 
                                    place in the graph.
    """
    # Initialize a count for both process
    t_grad_process = 0
    t_mix_process = 0

    wait_times_grad = []
    wait_times_mix = []

    # create the waiting times for the gradient process
    while t_grad_process < t_max:
        u = 1 - np.random.rand()  # samples of random.rand in [0, 1)
        t_grad = -np.log(u) / lamb_grad
        t_grad_process += t_grad
        wait_times_grad.append(t_grad)

    # create the waiting times for the mixing process
    while t_mix_process < t_max:
        u = 1 - np.random.rand()  # samples of random.rand in [0, 1)
        t_mix = -np.log(u) / lamb_mix
        t_mix_process += t_mix
        wait_times_mix.append(t_mix)

    # order the jumps: next jump is a gradient one or a mixing one ?
    take_grad_step = []
    t_event = []
    # k_grad and k_mix keep in memory the id of the last jump for
    # each process in the list of waiting times
    k_grad = 0
    k_mix = 0
    # compute the cumulative sum of the waiting times for both processes
    cumsum_grad = np.cumsum(wait_times_grad)
    cumsum_mix = np.cumsum(wait_times_mix)

    for i in range(len(wait_times_grad) + len(wait_times_mix)):
        is_grad = np.argmin([cumsum_mix[k_mix], cumsum_grad[k_grad]])
        if is_grad:
            t_event.append(cumsum_grad[k_grad])
            k_grad += 1
            take_grad_step.append(True)
        else:
            t_event.append(cumsum_mix[k_mix])
            k_mix += 1
            take_grad_step.append(False)
        # if any of the two k are too large, it means that we
        # already have a process longer than t_max, so we can stop
        if k_grad >= len(wait_times_grad) or k_mix >= len(wait_times_mix):
            break

    return take_grad_step, t_event


def create_random_connected_geometric_graph(n, radius):
    """ 
    Create a random geometric graph and add a minimal number of edges
    between connected components until the graph is connected.
    
    Parameters:
        - n (int): the number of nodes.
        - radius (float): radius used for the geometric graphs.
    
    Returns:
        - G_connected (nx.graph): a NetworkX connected graph.
    """
    # uniform sampling of n 2D points in [0,1]x[0,1]
    pos = {i: (np.random.random(), np.random.random()) for i in range(n)}
    # Create the geometric graph from the point cloud
    G = nx.random_geometric_graph(n, radius, pos=pos)
    is_connected = nx.is_connected(G)
    # If the graph is not connected
    if not is_connected:
        G_connected = G.copy()
        S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        # create an additional edge between two random nodes
        # in consecutive connected components
        for k in range(len(S) - 1):
            node_k = np.random.choice(S[k].nodes)
            node_kp = np.random.choice(S[k + 1].nodes)
            G_connected.add_edge(node_k, node_kp)
    else:
        G_connected = G

    return G_connected


def generate_one_connected_graph(n, graph_type, radius=None):
    """
    Generate one connected graph of a given size and of the specified type.
    
    Parameters:
        - n (int): number of nodes in the graph.
        - graph_type (str): type of the graph to generate.
                            We support either of ['star', 'complete', 'cycle', 'path', '2D_grid', 'barbell', 'random_geom']
        - radius (float): should be in (0, sqrt(2)),
                          the radius parameter to use for the geometric graphs.
    Returns:
        - G (nx.graph): A NetworkX connected graph.
    """

    if graph_type not in [
        "star",
        "complete",
        "cycle",
        "path",
        "2D_grid",
        "barbell",
        "random_geom",
    ]:
        raise ValueError(
            "The graph_type considered are ['star', 'complete', 'cycle', 'path', '2D_grid', 'barbell', 'random_geom']."
        )

    if graph_type == "star":
        G = nx.star_graph(n - 1)
    elif graph_type == "complete":
        G = nx.complete_graph(n)
    elif graph_type == "cycle":
        G = nx.cycle_graph(n)
    elif graph_type == "path":
        G = nx.path_graph(n)
    elif graph_type == "2D_grid":
        m = np.sqrt(n)
        if m != int(m):
            print(
                "WARNING: the number of nodes of the 2D grid will not be n, but in(sqrt(n))^2."
            )
        G = nx.grid_2d_graph(int(m), int(m))
        # convert the nodes id and the edges to integers and tuples of integers, respectively.
        G = nx.convert_node_labels_to_integers(G)
    elif graph_type == "barbell":
        m = (n) / 2
        if m != int(m):
            print(
                "WARNING: the number of nodes of the barbell will not be n, but 2*int(n/2)."
            )
        G = nx.barbell_graph(int(m), 0)
    elif graph_type == "random_geom":
        if radius is None:
            raise ValueError(
                "For the random geometric graph, a radius value should be given."
            )
        G = create_random_connected_geometric_graph(n, radius)

    return G


def create_K_graphs(n, graph_type, radius=None, K=50):
    """
    Create K graphs.
    
    Parameters:
        - n (int): the number of nodes in each graph.
        - graph_type (str): the graph type to generate.
        - radius (float): radius used for the geometric graphs.
        - K (int): number of graphs to produce.
        
    Returns:
        - list_G (list): list of K nx.Graph connected graphs.
        - list_W (list): list of the K Laplacian matrices used in ADOM+.
        - list_L_norm (list): list of the K Laplacian matrices used in the Continuized.
        - chi_1_star (float): worst case chi_1 of the K matrices.
        - chi_2_star (float): worst case chi_2 of the K matrices.
        - chi (float): worst case chi of the K matrices.
    """
    # In the random geometric graph case,
    # we have to create 50 different graph,
    # Thus it is necessary to compute the spectral quantities of each
    # to find the worst cases.
    if graph_type == "random_geom":
        # Initialize the lists
        list_G = []
        list_W = []
        list_L_norm = []
        list_chi_1 = []
        list_chi_2 = []
        list_chi = []
        for k in range(K):
            # Create 50 different connected graphs
            G = generate_one_connected_graph(n, graph_type, radius)
            W, L_norm, chi_1, chi_2, chi = compute_spectral_quantities(G)
            list_G.append(G)
            list_W.append(W)
            list_L_norm.append(L_norm)
            list_chi_1.append(chi_1)
            list_chi_2.append(chi_2)
            list_chi.append(chi)
        # compute the worst case over the 50 graphs.
        chi_1_star = max(list_chi_1)
        chi_2_star = max(list_chi_2)
        chi_star = max(list_chi)
    # If we are not in the random case, there is only 1 graph topology to consider.
    else:
        G = generate_one_connected_graph(n, graph_type, radius)
        W, L_norm, chi_1_star, chi_2_star, chi_star = compute_spectral_quantities(G)
        list_G = [G] * K
        list_W = [W] * K
        list_L_norm = [L_norm] * K

    return list_G, list_W, list_L_norm, chi_1_star, chi_2_star, chi_star
