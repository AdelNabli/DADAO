import torch
import torch.nn as nn


class logistic_regression(nn.Module):
    """
    Apply a l2 regularized logistic regression loss function, i.e, apply the following function for each worker i:
    
    f_i(\theta, X, y) = (1/m)*sum_{j=1}^m \log(1 + exp(-y_{ij} < x_{ij}, \theta >)) + \mu/2 || \theta ||^2
    
    where:
    
        - m is the size of the dataset at each worker
        - y_{ij} are the labels in {-1, 1}
        - x_{ij} are the data points
        - \theta are the parameters of the logistic regression
        - \mu is the strength of the regularization
    """

    def __init__(self, mu, dim, n_workers, n_data, centralized=False):
        super().__init__()
        """
        Initialize the logistic regression function depending on whether or not
        we are in the centralized or decentralized setting.
        
        Parameters:
            - mu (float): the strong convexity parameter
            - dim (int): the dimension of the data points
            - n_workers (int): the total number of workers
            - n_data (int): number of data points stored in each worker.
            - centralized (bool): whether or not to place ourselves
                                  in the centralized setting
        """

        self.mu = mu
        self.centralized = centralized
        self.n_data = n_data
        if centralized:
            # if centralized, we use one set of parameters: all the workers
            # share the same parameters. theta_i of shape [1, dim, 1]
            self.theta_i = nn.Parameter(
                torch.randn((1, dim, 1)).double(), requires_grad=True
            )
        else:
            # if decentralized, we have one different set of parameters for each worker.
            # theta_i of shape [n_workers, dim, 1]
            self.theta_i = nn.Parameter(
                torch.randn((n_workers, dim, 1)).double(), requires_grad=True
            )

    def forward(self, X, y):
        """
        Apply the functions f_i(\theta, X, y) and returns the collection of n_workers scalars.
        
        Parameters:
            - X (torch.tensor): of shape [n_workers, n_data_per_worker, dim],
                                contains the data points in each worker.
            - y (torch.tensor): of shape [n_workers, n_data_per_worker, dim],
                                contains the labels (in {-1,1}) of the data points.
        Returns:
            - out (torch.tensor): of shape [n_workers, 1],
                                  the values of f_i(\theta, X, y)
        """

        # apply the batch dot product between the parameters and the data
        if self.centralized:
            out = torch.matmul(
                X, self.theta_i
            )  # out of shape [n_workers, n_data_per_worker, 1]
        else:
            out = torch.bmm(
                X, self.theta_i
            )  # out of shape [n_workers, n_data_per_worker, 1]
        # multiply by the labels
        out = (-1 * y) * out
        # apply the logistic regression function
        out = torch.log(1 + torch.exp(out))
        # mean over the data in each worker
        out = torch.sum(out, dim=1) / self.n_data
        # add the regularizer
        out = out + (self.mu / 2) * torch.linalg.norm(self.theta_i, dim=1) ** 2

        return out


class linear_regression(nn.Module):
    """
    Apply the squared loss for linear regression, i.e, apply the following function for each worker i:
    
    f_i(\theta, X, y) = (1/m)*sum_{j=1}^m (< x_{ij},  \theta > - y_{ij})^2
    
    where:
    
        - m is the size of the dataset at each worker
        - y_{ij} are the targets values
        - x_{ij} are the data points
        - \theta are the parameters of the linear regression
    """

    def __init__(self, dim, n_workers, n_data, centralized=False):
        super().__init__()
        """
        Initialize the linear regression function depending on whether or not
        we are in the centralized or decentralized setting.
        
        Parameters:
            - dim (int): the dimension of the data points
            - n_workers (int): the total number of workers
            - n_data (int): number of data points stored in each worker.
            - centralized (bool): whether or not to place ourselves
                                  in the centralized setting
        """

        self.centralized = centralized
        self.n_data = n_data
        if centralized:
            # if centralized, we use one set of parameters: all the workers
            # share the same parameters. theta_i of shape [1, dim, 1]
            self.theta_i = nn.Parameter(
                torch.randn((1, dim, 1)).double(), requires_grad=True
            )
        else:
            # if decentralized, we have one different set of parameters for each worker.
            # theta_i of shape [n_workers, dim, 1]
            self.theta_i = nn.Parameter(
                torch.randn((n_workers, dim, 1)).double(), requires_grad=True
            )

    def forward(self, X, y):
        """
        Apply the functions f_i(\theta, X, y) and returns the collection of n_workers scalars.
        
        Parameters:
            - X (torch.tensor): of shape [n_workers, n_data_per_worker, dim],
                                contains the data points in each worker.
            - y (torch.tensor): of shape [n_workers, n_data_per_worker, dim],
                                contains the target values of each data point.
        Returns:
            - out (torch.tensor): of shape [n_workers, 1],
                                  the values of f_i(\theta, X, y)
        """

        # apply the batch dot product between the parameters and the data
        if self.centralized:
            out = torch.matmul(
                X, self.theta_i
            )  # out of shape [n_workers, n_data_per_worker, 1]
        else:
            out = torch.bmm(
                X, self.theta_i
            )  # out of shape [n_workers, n_data_per_worker, 1]
        # substract the target and square
        out = (out - y) ** 2
        # mean over the data in each worker
        # out = torch.sum(out, dim=1)/self.n_data
        out = torch.mean(out, dim=1)
        return out
