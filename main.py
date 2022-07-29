import argparse
import torch
import scipy
import datetime
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from cvx_functions import logistic_regression, linear_regression
from utils import save_data, compute_regularity_constants, dual_grad_linear_regression
from run_optimizers import run_optimizer
from data import create_data, create_K_graphs


def get_args_parser():

    parser = argparse.ArgumentParser("Distributed Optimization Script", add_help=False)
    parser.add_argument(
        "--optimizer_name",
        default="DADAO",
        type=str,
        help="Name of the optimizer to use. We support either one of ['DADAO', ADOMplus', 'MSDA', 'Continuized']",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Data points to use. Tensor of shape [n_workers, n_data_per_worker, dim].",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Labels of the data points. Tensor of shape [n_workers, n_data_per_worker, 1].",
    )
    parser.add_argument(
        "--classification",
        default=False,
        type=bool,
        help="Whether or not the task at hand is binary classification or linear regression.",
    )
    parser.add_argument(
        "--mu",
        default=1.0,
        type=float,
        help="Coefficient of strong convexity of the function to optimize.",
    )
    parser.add_argument(
        "--L",
        default=1.0,
        type=float,
        help="Coefficient of smoothness of the function to optimize.",
    )
    parser.add_argument(
        "--chi", default=1, type=float, help="Graph condition number for ADOM+."
    )
    parser.add_argument(
        "--chi_2", default=1, type=float, help="Graph's effective resistance."
    )
    parser.add_argument(
        "--chi_1",
        default=1,
        type=float,
        help="Inverse of the smallest positive eigenvalue of the Laplacian used in DADAO and Continuized.",
    )
    parser.add_argument(
        "--dim", default=2, type=int, help="Dimension of the datapoints."
    )
    parser.add_argument(
        "--n_workers", default=10, type=int, help="Number of workers to use."
    )
    parser.add_argument(
        "--t_max",
        default=100,
        type=int,
        help="Time interval during which we run the DADAO or Continuized optimizers. In ADOMplus and MSDA, number of synchronous gradient steps we take.",
    )
    parser.add_argument(
        "--lamb_grad",
        default=None,
        type=float,
        help="lambda parameter of the poisson point process for the gradient steps, should be equal to n_workers.",
    )
    parser.add_argument(
        "--lamb_mix",
        default=1.0,
        type=float,
        help="lambda parameter of the poisson point process for the gradient steps.",
    )
    parser.add_argument(
        "--list_G",
        type=list,
        default=None,
        help="List of the 50 connected nx.Graph to use for simulating the time varying nature of the connectivity network.",
    )
    parser.add_argument(
        "--list_W",
        type=list,
        default=None,
        help="List of the 50 corresponding normalized Laplacian matrices for ADOMplus.",
    )
    parser.add_argument(
        "--use_multi_consensus",
        default=False,
        type=bool,
        help="Whether or not run the Multi-Consensus version of ADOM+.",
    )
    parser.add_argument(
        "--stochastic",
        default=False,
        type=bool,
        help="Whether or not run DADAO with stochastic gradients.",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="If stochastic, the mini-batch size to use for the local sampling of data points.",
    )
    parser.add_argument(
        "--graph_type",
        default="complete",
        type=str,
        help="The type of graph to run the optimizer on to, we support either one of ['star', 'complete', 'cycle', 'path', '2D_grid', 'barbell', 'random_geom']",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Total number of samples in the aggregated dataset.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.4,
        help="Radius to use to create the random connected geometric graphs.",
    )

    return parser


def main(args):

    # Sanity check
    if args.list_W is None or args.list_G is None:
        raise ValueError(
            "A list of times varying connected graphs and corresponding normalized Laplacian matrices should be given."
        )
    if args.optimizer_name not in ["ADOMplus", "DADAO", "MSDA", "Continuized"]:
        raise ValueError(
            "We support either one of ['ADOMplus', 'DADAO', 'MSDA', 'Continuized'] optimizer."
        )
    if args.optimizer_name in ["MSDA", "Continuized"] and args.classification:
        raise ValueError(
            " For MSDA and the Continuized framework, we only support the Linear Regression task."
        )
    if (
        args.optimizer_name in ["MSDA", "Continuized"]
        and args.graph_type == "random_geom"
    ):
        raise ValueError(
            " For MSDA and the Continuized framework, we do not support time-varying graphs."
        )
    if args.data is None or args.labels is None:
        raise ValueError("A dataset should be given.")

    # INITIALIZE THE FUNCTION TO OPTIMIZE
    # if the task is the binary classification one
    if args.classification:
        # Initialize the function
        f = logistic_regression(args.mu, args.dim, args.n_workers, args.data.shape[1])
    # If the task is the linear regression
    else:
        # Initialize the function
        f = linear_regression(args.dim, args.n_workers, args.data.shape[1])

    # COMPUTE THE TRUE OPTIMAL VALUE WITH SKLEARN
    data_sklearn = torch.cat([args.data[k] for k in range(args.data.shape[0])]).numpy()
    labels_sklearn = (
        torch.cat([args.labels[k] for k in range(args.labels.shape[0])])
        .squeeze()
        .numpy()
    )
    if args.classification:
        clf = LogisticRegression(
            penalty="l2", fit_intercept=False, C=1 / (args.mu * len(data_sklearn))
        ).fit(data_sklearn, labels_sklearn)
        x_star = clf.coef_
        f_star = None
    else:
        x_star, res, _, _ = scipy.linalg.lstsq(data_sklearn, labels_sklearn)
        f_star = res / len(data_sklearn)
    x_star = x_star.ravel()

    # gather the date
    time = datetime.datetime.now()
    time_now = "_".join(
        [
            str(time_part)
            for time_part in [
                time.year,
                time.month,
                time.day,
                time.hour,
                time.minute,
                time.second,
            ]
        ]
    )

    # RUN THE DECENTRALIZED OPTIMIZER
    optimizer, loss_list, loss_list_edges = run_optimizer(args, f, x_star, time_now)

    # SAVES THE DATA FROM THE RUN
    save_data(
        loss_list,
        time_now,
        args.optimizer_name,
        args.use_multi_consensus,
        args.stochastic,
        args.graph_type,
        args.n_workers,
        is_loss_comp=True,
    )
    save_data(
        loss_list_edges,
        time_now,
        args.optimizer_name,
        args.use_multi_consensus,
        args.stochastic,
        args.graph_type,
        args.n_workers,
        is_loss_comp=False,
    )

    # PRINT THE FINAL INFORMATIONS
    with torch.no_grad():
        # primal methods
        if args.optimizer_name not in ["Continuized", "MSDA"]:
            # in order to compute f(x_bar)
            X = optimizer.X
        # dual methods
        else:
            if args.optimizer_name == "MSDA":
                opt_X = optimizer.X
            else:
                opt_X = optimizer.Y
            X = []
            for i in range(args.n_workers):
                X.append(
                    dual_grad_linear_regression(
                        args.data, args.labels, opt_X[i], i
                    ).unsqueeze(0)
                )
            X = torch.cat(X)
        X_bar = torch.mean(X, dim=0)
        f.centralized = True
        f.theta_i = nn.Parameter(X_bar.unsqueeze(-1))
        print(
            "f_star sklearn : ",
            f_star,
            "        | ",
            "f(x_bar) %s : " % args.optimizer_name,
            torch.mean(f(args.data, args.labels)).detach(),
        )
        print(
            "x_star sklearn : ",
            x_star,
            " | ",
            "x_bar %s : " % args.optimizer_name,
            X_bar.numpy(),
        )

    return optimizer, loss_list, loss_list_edges


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(
        "Distributed Optimization Script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.data is None or args.labels is None:
        # create a dataset
        args.data, args.labels = create_data(
            args.n_samples, args.dim, args.n_workers, args.classification
        )
        # compute the smoothness and strong convexity coefficients
        args.mu, args.L = compute_regularity_constants(args.data, args.classification)
    if args.list_G is None or args.list_W is None:
        # create a sequence of time varying connected graphs
        args.list_G, args.list_W, list_L_norm, args.chi_1, args.chi_2, args.chi = create_K_graphs(
            args.n_workers, args.graph_type, radius=args.radius, K=50
        )
        # compute the right intensity value for the mixing process
        args.lamb_mix = np.sqrt(2 * args.chi_1 * args.chi_2)
    if args.lamb_grad is None:
        # gives the right intensity value for the gradient process
        args.lamb_grad = args.n_workers
    # run main
    _, _, _ = main(args)
