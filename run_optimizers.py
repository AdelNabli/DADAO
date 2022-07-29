import torch
import numpy as np
from tqdm import trange
from data import create_asynchronous_jump_processes
from utils import (
    save_data,
    compute_average_distance_to_opt,
    dual_grad_linear_regression,
)
from optimizers import (
    DADAO_optimizer,
    ADOMplus_optimizer,
    Continuized_optimizer,
    MSDA_optimizer,
)


def run_DADAO(
    n_workers,
    t_max,
    lamb_grad,
    lamb_mix,
    chi_1,
    f,
    mu,
    L,
    data,
    labels,
    x_star,
    list_G,
    stochastic,
    batch_size,
    time_now,
    graph_type,
):
    """
    Run our asynchronous decentralized optimizer procedure by simulating in advance
    a sequence of random events corresponding either to one gradient step
    on one node or a communication along an edge.
    
    Parameters:
        - n_workers (int): the number of workers
        - t_max (float): the duration of the run
        - lamb_grad (float): the intensity of the poisson process for the
                           gradient steps at the graph's scale.
                           Should be = n_workers
        - lamb_mix (float): the intensity of the poisson process for the
                            communication steps at the graph's scale.
                            Should be = sqrt{2*chi_1_star*chi_2_star}
        - chi_1 (float): the chi_1 of the Laplacian used in the Continuized framework.
        - f (torch.nn.Module): the convex function to optimize.
        - mu (float): the strong convexity coefficient
        - L (float): the smoothness coefficient.
        - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim],
                               the data stored on each worker.
        - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1],
                                  the labels.
        - x_star (np.array): of size dim.
                             The optimal parameter of the corresponding centralized
                             problem (found using scikit learn).
        - list_G (list): the list of time-varying graphs on which we cycle through.
        - stochastic (bool): whether ir not to use stochastic gradients (by sampling a minibatch locally)
        - batch_size (int): if stochastic, the size of the minibatch to use.
        - time_now (str): the time of the run (only used to name the file storing the run's data.)
        - graph_type (str): the nature of the graphs.
        
    Returns:
        - optimizer (DADAO optimizer): the optimizer object.
        - loss_list (list): the evolution of the average distance to x^*
                            at each individual gradient step.
        - loss_list_edges (list): the evolution of the average distance to x^*
                                  at each individual communication.
    """
    # INITIALIZATION
    # initialize the optimizer
    # The chi_1 of the Laplacian \Lambda is the chi_1
    # of the Laplacian of probas divided by the intensity
    # of the P.P.P for communication
    chi_1_star = chi_1 / lamb_mix
    optimizer = DADAO_optimizer(
        f,
        data,
        labels,
        chi_1_star=chi_1_star,
        lamb_mix=lamb_mix,
        n_nodes=n_workers,
        mu=mu,
        L=L,
        stochastic=stochastic,
        batch_size=batch_size,
    )
    # create the random sequence of events
    is_grad_step, list_t_event = create_asynchronous_jump_processes(
        t_max, lamb_grad, lamb_mix
    )
    # initialize the list of losses
    loss_list = []
    loss_list_edges = []
    # In the stochastic case, keep track of the old parameters
    # to perform a running average.
    # Initialize X_all, a variable storing them and a local
    # count keeping track of the number of local changes occured at each node.
    X_all = torch.zeros(optimizer.X.shape).double()
    # use eps to prevent from dividing by zero in the first few steps.
    eps = 1e-24
    counts_nodes = eps * torch.ones(n_workers).double().unsqueeze(-1)

    # RUN THE OPTIMIZATION PROCEDURE
    # play all the events in order of appearance
    for k in trange(len(list_t_event)):
        # if event k is a grad step
        if is_grad_step[k]:
            # chose a node to update uniformly at random
            node_i = np.random.randint(0, n_workers)
            edge_ij = None
            # for the running average in the stochastic case
            X_all[node_i] = X_all[node_i] + optimizer.X[node_i]
            counts_nodes[node_i] = counts_nodes[node_i] + 1
        else:
            # We change of graph every 1/chi unit of time as in ADOM+
            # with multi-consensus
            G = list_G[int(list_t_event[k]*chi_1) % len(list_G)]
            # choose an edge uniformly at random among the edges
            # of the connected graph we consider at this time
            edge_ij = np.array(G.edges)[np.random.choice(len(G.edges))]
            node_i, node_j = edge_ij
            # for the running average in the stochastic case
            X_all[node_i] = X_all[node_i] + optimizer.X[node_i]
            counts_nodes[node_i] = counts_nodes[node_i] + 1
            X_all[node_j] = X_all[node_j] + optimizer.X[node_j]
            counts_nodes[node_j] = counts_nodes[node_j] + 1
        # take an optimization step
        optimizer.step(is_grad_step[k], list_t_event[k], edge_ij, node_i)

        # SAVE THE DATA FROM THE RUN
        if stochastic:
            # compute the time average
            X_mean = X_all / counts_nodes
            loss = compute_average_distance_to_opt(X_mean, x_star)
        if not stochastic:
            # We compute the distance to optimal params every time
            loss = compute_average_distance_to_opt(optimizer.X, x_star)
        if is_grad_step[k]:
            loss_list.append(loss)
        if not is_grad_step[k]:
            loss_list_edges.append(loss)
        # save the data from the runs every 10 000 events
        if k % 10000 == 0:
            save_data(
                loss_list,
                time_now,
                "DADAO",
                False,
                stochastic,
                graph_type,
                n_workers,
                is_loss_comp=True,
            )
            save_data(
                loss_list_edges,
                time_now,
                "DADAO",
                False,
                stochastic,
                graph_type,
                n_workers,
                is_loss_comp=False,
            )

    return optimizer, loss_list, loss_list_edges


def run_ADOMplus(
    n_workers,
    n_steps,
    chi,
    f,
    mu,
    L,
    data,
    labels,
    x_star,
    list_G,
    list_W,
    use_multi_consensus,
    time_now,
    graph_type,
):
    """
    Run the ADOM+ optimizer (see Kovalev et al. https://openreview.net/forum?id=L8-54wkift )
    
    Parameters:
        - n_workers (int): the number of workers
        - n_steps (int): the number of synchronized gradient steps to perform.
        - chi (float): the chi of the Laplacian used in ADOM+.
        - f (torch.nn.Module): the convex function to optimize.
        - mu (float): the strong convexity coefficient
        - L (float): the smoothness coefficient.
        - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim],
                               the data stored on each worker.
        - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1],
                                  the labels.
        - x_star (np.array): of size dim.
                             The optimal parameter of the corresponding centralized
                             problem (found using scikit learn).
        - list_G (list): the list of time-varying graphs on which we cycle through.
        - list_W (list): list of gossip matrices corresponding to the graphs G.
        - use_multi_consensus (bool): whether or not use the multi consensus inner loop.
        - time_now (str): the time of the run (only used to name the file storing the run's data.)
        - graph_type (str): the nature of the graphs.
        
    Returns:
        - optimizer (ADOM+ optimizer): the optimizer object.
        - loss_list (list): the evolution of the average distance to x^*
                            at each individual gradient step.
        - loss_list_edges (list): the evolution of the average distance to x^*
                                  at each individual communication.
    """

    optimizer = ADOMplus_optimizer(f, data, labels, mu=mu, L=L, chi=chi)
    # Initialization of the optimization procedure
    k_mixing = 0
    In = torch.eye(n_workers).double()
    loss_list = []
    loss_list_edges = []
    if use_multi_consensus:
        n_matrix = int(np.ceil(chi * np.log(2)))
        # if we use multi-consensus, it amounts to changing the
        # effective chi of the communication matrices used.
        # Equation 20 of ( https://openreview.net/forum?id=L8-54wkift )
        # shows that the target value of chi used is 2 for M.C.
        # We thus change the value of chi in the optimizer,
        # only if n_matrix > 1
        if n_matrix > 1:
            optimizer.chi = 2
            optimizer.initialize()
    else:
        n_matrix = 1
    # Run ADOM+
    for k in trange(n_steps):
        # compute the multi-consensus matrix
        W_final = torch.eye(n_workers).double()
        n_edges = 0
        for q in range(n_matrix):
            # apply the procedure described in eq. 18 of the paper.
            W_final = W_final @ (In - list_W[k_mixing % len(list_W)])
            G = list_G[k_mixing % len(list_G)]
            n_edges += len(G.edges)
            k_mixing += 1
        # take a gradient step
        optimizer.step(W_final)
        # compute distance to optimal params
        with torch.no_grad():
            loss = compute_average_distance_to_opt(optimizer.X, x_star)
            loss_list = (
                loss_list + [loss] * n_workers
            )  # we take n gradients at each round
            loss_list_edges = (
                loss_list_edges + [loss] * n_edges
            )  # n_edges counted the total number of edges activated
        # regularly save the data from the runs
        if k % 10000 == 0:
            save_data(
                loss_list,
                time_now,
                "ADOM+",
                use_multi_consensus,
                False,
                graph_type,
                n_workers,
                is_loss_comp=True,
            )
            save_data(
                loss_list_edges,
                time_now,
                "ADOM+",
                use_multi_consensus,
                False,
                graph_type,
                n_workers,
                is_loss_comp=False,
            )

    return optimizer, loss_list, loss_list_edges


def run_continuized(
    n_workers,
    t_max,
    f,
    mu,
    L,
    mu_gossip,
    chi_2,
    data,
    labels,
    x_star,
    list_G,
    time_now,
    graph_type,
):
    """
    Run the Continuized optimizer of Event et al. https://arxiv.org/pdf/2106.07644.pdf,
    in the case of decentralized optimization (Appendix H., page 30)
    
    Parameters:
        - n_workers (int): the number of workers.
        - t_max (float): the duration of the run.
        - f (torch.nn.Module): the convex function to optimize.
        - mu (float): the strong convexity coefficient.
        - L (float): the smoothness coefficient.
        - mu_gossip (float): the smallest positive eigenvalue of the Laplacian considered.
        - chi_2 (float): the graph's resistance (computed with the Laplacian of probas)
        - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim],
                               the data stored on each worker.
        - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1],
                                  the labels.
        - x_star (np.array): of size dim.
                             The optimal parameter of the corresponding centralized
                             problem (found using scikit learn).
        - list_G (list): the list of time-varying graphs on which we cycle through.
        - time_now (str): the time of the run (only used to name the file storing the run's data.)
        - graph_type (str): the nature of the graphs.
        
    Returns:
        - optimizer (ADOM+ optimizer): the optimizer object.
        - loss_list (list): the evolution of the average distance to x^*
                            at each individual gradient step.
        - loss_list_edges (list): the evolution of the average distance to x^*
                                  at each individual communication.
    """

    optimizer = Continuized_optimizer(
        data,
        labels,
        mu=mu,
        L=L,
        mu_gossip=mu_gossip,
        chi_2=chi_2,
        G=list_G[0],  # fixed topology
    )
    # Create the list of waiting times for the poisson jumps
    wait_times = []
    t_mix_process = 0
    # the waiting times between two jumps follow a Poisson law
    # of parameter 1.
    while t_mix_process < t_max:
        u = 1 - np.random.rand()  # samples of random.rand in [0, 1)
        t_mix = -np.log(u) / 1  # a rate of 1 is used in Event et al.
        t_mix_process += t_mix
        wait_times.append(t_mix)
    # the times of the events is the cumulative sum of the waiting times.
    list_t_event = np.cumsum(wait_times)
    optimizer.times = list_t_event
    # Initialize the count of communication steps.
    k_mixing = 0
    # initialize the list of losses
    loss_list = []
    loss_list_edges = []
    # play all the events in order of appearance
    for k in trange(len(list_t_event)):
        # choose an edge uniformly at random among the edges
        # of the connected graph we consider at this time
        G = list_G[k_mixing % len(list_G)]
        edge_ij = np.array(G.edges)[np.random.choice(len(G.edges))]
        k_mixing += 1
        optimizer.step(list_t_event[k], edge_ij)
        # We compute the distance to optimal params every time
        with torch.no_grad():
            # gather the dual gradients of all workers
            X = []
            for i in range(n_workers):
                X.append(
                    dual_grad_linear_regression(
                        data, labels, optimizer.Z[i], i
                    ).unsqueeze(0)
                )
            X = torch.cat(X)
            loss = compute_average_distance_to_opt(X, x_star)
            loss_list_edges.append(loss)
            loss_list = (
                loss_list + [loss] * 2
            )  # we take 2 gradient steps for each communication
        # regularly save the data from the runs
        if k % 10000 == 0:
            save_data(
                loss_list,
                time_now,
                "Continuized",
                False,
                False,
                graph_type,
                n_workers,
                is_loss_comp=True,
            )
            save_data(
                loss_list_edges,
                time_now,
                "Continuized",
                False,
                False,
                graph_type,
                n_workers,
                is_loss_comp=False,
            )

    return optimizer, loss_list, loss_list_edges


def run_MSDA(
    n_workers, n_steps, f, mu, L, data, labels, x_star, list_G, time_now, graph_type
):
    """
    Run the Multi-Step Dual Accelerated (MSDA) method
    from Scaman et al https://arxiv.org/pdf/1702.08704.pdf
    
    Parameters:
        - n_workers (int): the number of workers.
        - n_steps (int): the number of synchronized gradient steps to perform.
        - f (torch.nn.Module): the convex function to optimize.
        - mu (float): the strong convexity coefficient.
        - L (float): the smoothness coefficient.
        - data (torch.tensor): of shape [n_workers, n_data_per_worker, dim],
                               the data stored on each worker.
        - labels (torch.tensor): of shape [n_workers, n_data_per_worker, 1],
                                  the labels.
        - x_star (np.array): of size dim.
                             The optimal parameter of the corresponding centralized
                             problem (found using scikit learn).
        - list_G (list): the list of time-varying graphs on which we cycle through.
        - time_now (str): the time of the run (only used to name the file storing the run's data.)
        - graph_type (str): the nature of the graphs.
        
    Returns:
        - optimizer (ADOM+ optimizer): the optimizer object.
        - loss_list (list): the evolution of the average distance to x^*
                            at each individual gradient step.
        - loss_list_edges (list): the evolution of the average distance to x^*
                                  at each individual communication.
    """

    optimizer = MSDA_optimizer(data, labels, mu=mu, L=L, G=list_G[0])  # fixed topology
    # Initialize the count of gradient steps
    k_mixing = 0
    # initialize the list of losses
    loss_list = []
    loss_list_edges = []
    # retrieve the number of iteration in the Accelerated Gossip procedure
    K_msda = optimizer.K
    # compute the total number of edges fired for one gradient round
    n_edges_fired = len(list_G[0].edges) * K_msda
    for k in trange(n_steps):
        optimizer.step()
        k_mixing += 1
        # We compute the distance to optimal params every time
        with torch.no_grad():
            X = []
            for i in range(n_workers):
                X.append(
                    dual_grad_linear_regression(
                        data, labels, optimizer.X[i], i
                    ).unsqueeze(0)
                )
            X = torch.cat(X)
            loss = compute_average_distance_to_opt(X, x_star)
            loss_list_edges = loss_list_edges + [loss] * n_edges_fired
            loss_list = (
                loss_list + [loss] * n_workers
            )  # we take n gradient steps for one iteration
        # regularly save the data from the runs
        if k % 10000 == 0:
            save_data(
                loss_list,
                time_now,
                "MSDA",
                False,
                False,
                graph_type,
                n_workers,
                is_loss_comp=True,
            )
            save_data(
                loss_list_edges,
                time_now,
                "MSDA",
                False,
                False,
                graph_type,
                n_workers,
                is_loss_comp=False,
            )

    return optimizer, loss_list, loss_list_edges


def run_optimizer(args, f, x_star, time_now):
    """
    Helper to the main function to run the right optimizer from the arguments passed.
    """
    optimizer, loss_list, loss_list_edges = None, None, None
    if args.optimizer_name == "DADAO":
        optimizer, loss_list, loss_list_edges = run_DADAO(
            args.n_workers,
            args.t_max,
            args.lamb_grad,
            args.lamb_mix,
            args.chi_1,
            f,
            args.mu,
            args.L,
            args.data,
            args.labels,
            x_star,
            args.list_G,
            args.stochastic,
            args.batch_size,
            time_now,
            args.graph_type,
        )
    elif args.optimizer_name == "ADOMplus":
        optimizer, loss_list, loss_list_edges = run_ADOMplus(
            args.n_workers,
            int(args.t_max),
            args.chi,
            f,
            args.mu,
            args.L,
            args.data,
            args.labels,
            x_star,
            args.list_G,
            args.list_W,
            args.use_multi_consensus,
            time_now,
            args.graph_type,
        )
    elif args.optimizer_name == "Continuized":
        optimizer, loss_list, loss_list_edges = run_continuized(
            args.n_workers,
            args.t_max,
            f,
            args.mu,
            args.L,
            1 / args.chi_1,
            args.chi_2,
            args.data,
            args.labels,
            x_star,
            args.list_G,
            time_now,
            args.graph_type,
        )
    elif args.optimizer_name == "MSDA":
        optimizer, loss_list, loss_list_edges = run_MSDA(
            args.n_workers,
            int(args.t_max),
            f,
            args.mu,
            args.L,
            args.data,
            args.labels,
            x_star,
            args.list_G,
            time_now,
            args.graph_type,
        )
    return optimizer, loss_list, loss_list_edges
