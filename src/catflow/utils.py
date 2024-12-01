import math
import torch
import shutil
from datetime import datetime
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops

""" NOTE: The code is mostly taken from the DiGress repository: https://github.com/cvignac/DiGress unless stated otherwise. """


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def encode_no_edge(E):
    assert len(E.shape) == 4, "Expected shape [batch, nodes, nodes, edge_classes]"
    if E.shape[-1] == 0:
        return E

    # Find locations where no edge is present in any class
    no_edge = torch.sum(E, dim=3) == 0  # Shape: [batch, nodes, nodes]

    # Set absence indicator (channel 0) at no-edge locations symmetrically
    E[:, :, :, 0][no_edge] = 1
    E[:, :, :, 0] = torch.max(
        E[:, :, :, 0], E[:, :, :, 0].transpose(1, 2)
    )  # Make channel 0 symmetric

    # Copy all channels symmetrically for each [i, j] and [j, i] pair
    for k in range(E.shape[-1]):
        E[:, :, :, k] = torch.max(E[:, :, :, k], E[:, :, :, k].transpose(1, 2))

    # Set diagonal elements to zero for all channels
    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0

    # Ensure final symmetry in all channels
    assert torch.allclose(
        E, E.transpose(1, 2)
    ), "encode_no_edge produced a non-symmetric tensor"
    return E


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    node_mask = node_mask.float()
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    max_num_nodes = X.size(1)
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


# Taken from Davis et al. (2024) Fisher Flow Matching, https://github.com/olsdavis/fisher-flow
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def sample_gaussian(size):
    x = torch.randn(size)
    return x


def sample_feature_noise(X_size, E_size, y_size, node_mask):
    """Standard normal noise for all features.
    Output size: X.size(), E.size(), y.size()"""
    # TODO: How to change this for the multi-gpu case?
    epsX = sample_gaussian(X_size)
    epsE = sample_gaussian(E_size)
    epsy = sample_gaussian(y_size)

    float_mask = node_mask.float()
    epsX = epsX.type_as(float_mask)
    epsE = epsE.type_as(float_mask)
    epsy = epsy.type_as(float_mask)

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(epsE)
    indices = torch.triu_indices(row=epsE.size(1), col=epsE.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    epsE = epsE * upper_triangular_mask
    epsE = epsE + torch.transpose(epsE, 1, 2)

    assert (epsE == torch.transpose(epsE, 1, 2)).all()

    return PlaceHolder(X=epsX, E=epsE, y=epsy).mask(node_mask)


def sample_normal(mu_X, mu_E, mu_y, sigma, node_mask):
    """Samples from a Normal distribution."""
    eps = sample_feature_noise(
        mu_X.size(), mu_E.size(), mu_y.size(), node_mask
    ).type_as(mu_X)
    X = mu_X + sigma * eps.X
    E = mu_E + sigma.unsqueeze(1) * eps.E
    y = mu_y + sigma.squeeze(1) * eps.y
    return PlaceHolder(X=X, E=E, y=y)


# From https://github.com/ccr-cheng/statistical-flow-matching
def sample_simplex(*sizes, device="cpu", eps=1e-4):
    """
    Uniformly sample from a simplex.
    :param sizes: sizes of the Tensor to be returned
    :param device: device to put the Tensor on
    :param eps: small float to avoid instability
    :return: Tensor of shape sizes, with values summing to 1
    """
    x = torch.empty(*sizes, device=device, dtype=torch.float).exponential_(1)
    p = x / x.sum(dim=-1, keepdim=True)
    p = p.clamp(eps, 1 - eps)
    return p / p.sum(dim=-1, keepdim=True)


def sample_discrete_feature_noise(limit_dist, node_mask):
    """Sample from the limit distribution of the diffusion process"""
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    y_limit = limit_dist.y[None, :].expand(bs, -1)
    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_y = torch.empty((bs, 0))

    long_mask = node_mask.long()
    U_X = U_X.type_as(long_mask)
    U_E = U_E.type_as(long_mask)
    U_y = U_y.type_as(long_mask)

    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = U_E + torch.transpose(U_E, 1, 2)

    assert (U_E == torch.transpose(U_E, 1, 2)).all()

    return PlaceHolder(X=U_X, E=U_E, y=U_y).mask(node_mask)


def get_writer(log_dir: str = f"logs/{datetime.now()}") -> SummaryWriter:
    """
    Create a SummaryWriter object to log the training process.

    Args:
        log_dir (str): The directory to save the logs.

    Returns:
        SummaryWriter: The SummaryWriter object.
    """
    # Remove (potentially already existing) plots from the TensorBoard
    try:
        shutil.rmtree(log_dir)
    except FileNotFoundError:
        pass
    # Create a SummaryWriter object
    writer = SummaryWriter(log_dir)

    return writer

def get_writer_windows(log_dir: str = f"logs/{datetime.now()}") -> SummaryWriter:
    """
    Create a SummaryWriter object to log the training process.

    Args:
        log_dir (str): The directory to save the logs.

    Returns:
        SummaryWriter: The SummaryWriter object.
    """
    # Remove colons from log directory
    log_dir = log_dir.replace(':', '')

    # Remove (potentially already existing) plots from the TensorBoard
    try:
        shutil.rmtree(log_dir)
    except FileNotFoundError:
        pass
    # Create a SummaryWriter object
    writer = SummaryWriter(log_dir)

    return writer