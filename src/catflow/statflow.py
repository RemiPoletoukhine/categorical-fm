import einops
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

# from torchdiffeq import odeint
from functools import partial
from zuko.utils import odeint

from src.qm9 import qm9_dataset
import src.catflow.utils as utils
from src.catflow.transformer_model import GraphTransformer
from src.qm9.extra_features_molecular import ExtraMolecularFeatures


from src.catflow.utils import (
    timestep_embedding,
    to_dense,
    get_device,
    get_writer,
    EarlyStopper,
    PlaceHolder,
)
# Code taken directly from https://github.com/ccr-cheng/statistical-flow-matching/blob/main/models/categorical.py#L30
# and then adapted for the DiGress Graph transformer

try:
    from torch.func import vjp, jacrev, vmap  # torch >= 2.0
except ImportError:
    try:
        from functorch import vjp, jacrev, vmap  # 1.10 <= torch <= 1.13
    except ImportError:
        raise ImportError(
            '`functorch` not found! Update PyTorch to at least >= 1.10 AND install `functorch`. '
            'Note that in some early versions, `functorch` is not automatically installed with PyTorch. '
            'Otherwise, use PyTorch >= 2.0 with `torch.func` now shipped with PyTorch.'
        )

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    print('[WARNING] scipy not installed, OT not available')
    linear_sum_assignment = None



def sample_simplex(*sizes, device='cpu', eps=1e-4):
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


class GraphStatFlow(nn.Module, ABC):
    """
    Base class for categorical flow models.
    The model follows the Riemannian flow matching framework to learn a vector field
    on different induced Riemannian geometries on the probability simplex.

    :param encoder: encoder model
    :param data_dims: dimensions of the data points
    # Above is adapted to have n_classes for X,E,y
    :param n_class_nodes: number of classes for the nodes
    :param n_class_edges: number of classes for the edges
    # Above is adapted to have n_classes for X,E,y
    :param max_t: maximum timestep for the flow
    :param ot: whether to use optimal transport during training
    :param eps: small float to avoid instability
    """

    def __init__(self, config, data_infos, domain_features, device, max_t=1., ot=False, eps=1e-4):
        super().__init__()
        self.config = config
        self.device = device
        self.encoder = GraphTransformer(
            n_layers=config["n_layers"],
            input_dims=data_infos.input_dims,
            hidden_mlp_dims=config["hidden_mlp_dims"],
            hidden_dims=config["hidden_dims"],
            output_dims=data_infos.output_dims,
        ).to(self.device)
        max_n_nodes = int(data_infos.max_n_nodes)
        self.data_dims = [[max_n_nodes,], [max_n_nodes, max_n_nodes]]
        # self.n_class_nodes = data_infos.node_types
        # self.n_class_edges = data_infos.edge_types
        self.n_class = [data_infos.output_dims[p] for p in ['X','E','y']]
        self.domain_features = domain_features
        # self.total_data_dim = [torch.LongTensor(data_dim).prod().item() for data_dim in self.data_dims]
        self.total_data_dim = [torch.LongTensor(data_dim).prod().item() for data_dim in self.data_dims]
        self.max_t = max_t
        self.ot = ot
        self.eps = eps
        self._v = None  # for Hutchinson's trace estimator

    #################################################
    # Helper functions                              #
    #################################################

    @staticmethod
    def sample_prior(*size, device, eps=1e-4):
        """
        Sample from the prior noise distribution on the simplex.
        :param size: sizes of the Tensor to be returned
        :param device: device to put the Tensor on
        :param eps: small float to avoid instability
        :return: Tensor of shape size, with values summing to 1
        """
        return sample_simplex(*size, device=device, eps=eps)

    @staticmethod
    def prior_logp0(p0, eps=1e-4):
        """
        Compute the log probability of the prior noise distribution on the probability simplex.
        :param p0: noise samples, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: log probability of the prior noise distribution, Tensor of shape (...)
        """
        return 0

    @classmethod
    def sample_simplex_linear(cls, p, t, eps=1e-4):
        """
        Sample from a simplex with linearly decreasing L2 distance
        :param p: one-hot probability vector, Tensor of shape (..., n)
        :param t: timestep between 0 and 1
        :param eps: small float to avoid instability
        :return:
            p_bar: sampled values, same size as p, with values summing to 1
            log_prob: log probability of the sampled values
        """
        samples = sample_simplex(*p.size(), device=p.device, eps=eps)
        p_bar = samples * (1 - t) + p * t
        log_prob = cls.prior_logp0(samples, eps) - np.log(1 - t) * (p.size(-1) - 1)
        return p_bar, log_prob

    @staticmethod
    def preprocess(p):
        """
        Preprocess the data point of the simplex.
        :param p: data point on the simplex
        :return: preprocessed data point of the same shape
        """
        return p

    @staticmethod
    def postprocess(p_hat):
        """
        Postprocess the data point to get back to the simplex.
        :param p_hat: processed data point
        :return: data point on the simplex of the same shape
        """
        return p_hat

    #################################################
    # Riemannian manifold operations                #
    #################################################

    @classmethod
    def proj_x(cls, x, eps=0.):
        """
        Project the data point to the manifold.
        :param x: data point
        :param eps: small float to avoid instability
        :return: projected data point of the same shape
        """
        return x

    @classmethod
    def proj_vf(cls, vf, pt):
        """
        Project the vector field to the tangent space at a specific data point.
        :param vf: vector field
        :param pt: data point at which the tangent space is defined
        :return: projected vector field of the same shape
        """
        return vf

    @classmethod
    @abstractmethod
    def dist(cls, p, q, eps=0.):
        """
        Geodesic distance between two data points.
        :param p: first data point, Tensor of shape (..., n)
        :param q: second data point, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: distance between p and q, Tensor of shape (...)
        """
        pass

    @classmethod
    @abstractmethod
    def norm2(cls, p, u, eps=0.):
        """
        Squared Riemannian norm of a vector field in the tangent space.
        :param p: data point at which the tangent space is defined, Tensor of shape (..., n)
        :param u: vector field, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: squared Riemannian norm of u at p, Tensor of shape (...)
        """
        pass

    @classmethod
    @abstractmethod
    def exp(cls, p, u, eps=0.):
        """
        Exponential map exp_p(u).
        :param p: data point at which the tangent space is defined, Tensor of shape (..., n)
        :param u: vector field, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: exponential map, Tensor of shape (..., n)
        """
        pass

    @classmethod
    @abstractmethod
    def log(cls, p, q, eps=0.):
        """
        Logarithmic map log_p(q).
        :param p: data point at which the tangent space is defined, Tensor of shape (..., n)
        :param q: target data point, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: logarithmic map, Tensor of shape (..., n)
        """
        pass

    @classmethod
    def interpolate(cls, p, q, t, eps=0.):
        """
        Interpolate between two data points.
        :param p: source data point, Tensor of shape (..., n)
        :param q: target data point, Tensor of shape (..., n)
        :param t: timestep between 0 and 1, Tensor of shape (...)
        :param eps: small float to avoid instability
        :return: interpolant at timestep t, Tensor of shape (..., n)
        """
        return cls.exp(p, t * cls.log(p, q, eps), eps)

    @classmethod
    def vecfield(cls, p, q, t, eps=0.) -> tuple[Tensor, Tensor]:
        """
        Vector field at timestep t.
        :param p: source data point, Tensor of shape (..., n)
        :param q: target data point, Tensor of shape (..., n)
        :param t: timestep between 0 and 1, Tensor of shape (...)
        :param eps: small float to avoid instability
        :return:
            pt: interpolant at timestep t, Tensors of shape (..., n)
            vf: vector field, Tensors of shape (..., n)
        """

        t_reshaped = t.view(-1, *([1]*(len(p.size())-1)))
        pt = cls.interpolate(p, q, t_reshaped, eps)

        # print(pt.size(), 'vecfield')

        vf = cls.log(pt, q, eps) / (1 - t_reshaped)
        return pt, vf

    #################################################
    # Forward and loss functions                    #
    #################################################


    # Added utility function for the forward
    def pre_forward(self, t, pt, node_mask):

        # Instantiate a noisy data dict needed for computations
        t_expanded = t.expand(pt.X.shape[0])
        noisy_data = {"X_t": pt.X, "E_t": pt.E, "y_t": pt.y, "t": t_expanded, "node_mask": node_mask}

        # print(pt.X.size(), "pre_forward  pt")
        # print(noisy_data["X_t"].size(), "pre_forward  noisy")

        # Compute extra data, same as in the functuion in catflow.py "compute_extra_data"
        extra_data = self.domain_features(noisy_data)

        ty = noisy_data["t"].unsqueeze(-1)
        extra_data.y = torch.cat((extra_data.y, ty), dim=1)

        x_extra = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        e_extra = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y_extra = torch.hstack((noisy_data["y_t"], extra_data.y)).float()

        # embed the timestep using the sinusoidal positional encoding
        t_embedded_nodes = timestep_embedding(
            t, dim=x_extra.shape[-1]
        )  # Shape: (batch_size, num_classes_nodes)
        t_embedded_edges = timestep_embedding(
            t, dim=e_extra.shape[-1]
        )  # Shape: (batch_size, num_classes_edges)

        # print(t_embedded_nodes.size(), "t_embedded_nodes")
        # print(t_embedded_edges.size(), "t_embedded_edges")
        # add time embedding to the input across the class feature dimension
        x_extra += einops.rearrange(t_embedded_nodes, "b c -> b 1 c")
        e_extra += einops.rearrange(t_embedded_edges, "b c -> b 1 1 c")


        return PlaceHolder(X=x_extra, E=e_extra, y=y_extra)


    def forward(self, t, pt, node_mask, *cond_args):
        """
        predict vector field at time t
        :param t: timestep between 0 and 1, Tensor of shape (B,) or a single float (Changed to list of tensors of shape (B,))
        :param pt: interpolant at timestep t, Tensor of shape (B, D, n) (Changed to PlaceHolder object of interpolants)
        :return: predicted vector field, Tensor of shape (B, D, n)
        """
        # for t in ts:
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Interpolant with molecular features is returned
        pt_mol = self.pre_forward(t, pt, node_mask)
        pt_mol.mask(node_mask)
        ph_vf = self.encoder(pt_mol.X, pt_mol.E, pt_mol.y, node_mask)
        # print(ph_vf.X.size())
        # print(ph_vf.E.size())
        vf = [ph_vf.X.view(-1, self.total_data_dim[0], self.n_class[0]),
            ph_vf.E.view(-1, self.total_data_dim[1], self.n_class[1])]

        # Projections onto the tangent space at the interpolant pt for X and E respectively
        return self.proj_vf(vf[0], pt), self.proj_vf(vf[1], pt) 

    def batch_ot(self, p, q, cond_args):
        """
        Mini-batch optimal transport
        :param p: batch of noise samples, Tensor of shape (B, D, n)
        :param q: batch of data samples, Tensor of shape (B, D, n)
        :param cond_args: batch of conditional arguments, assume first dimension is the batch dimension
        :return: OT-matched noise samples, data samples, and conditional arguments
        """
        dist = self.dist(p.unsqueeze(1), q.unsqueeze(0), 0.).sum(-1)
        noise_idx, data_idx = linear_sum_assignment(dist.cpu().numpy())
        return p[noise_idx], q[data_idx], [cond[data_idx] for cond in cond_args]

    def get_loss(self, data, logger, device, *cond_args):
        """
        Compute the Riemannian flow matching loss.
        :param data: the data from the dataloader
        :param cond_args: optional conditional arguments
        :return: loss
        """

        if data.edge_index.numel() == 0:
            logger.info("Found a batch with no edges. Skipping.")
            return
        data = data.to(device)
        # Get the dense representation of the graph
        dense_data, node_mask = to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask)
        x_1, e_1, node_mask = (
            dense_data.X.to(device),
            dense_data.E.to(device),
            node_mask.to(device),
        )
        y_1 = data.y.to(device)


        self.data_dims[0] = [x_1.size()[1:-1][0],]
        self.data_dims[1] = list(e_1.size()[1:-1])
        self.total_data_dim = [torch.LongTensor(data_dim).prod().item() for data_dim in self.data_dims]


        vf = [0,0]
        t = torch.rand(x_1.size(0), device=device) * self.max_t


        noise = PlaceHolder(X=self.sample_prior(x_1.size()[0], *self.data_dims[0], self.n_class[0], device=device), 
                            E=self.sample_prior(e_1.size()[0], *self.data_dims[1], self.n_class[1], device=device), 
                            y=y_1)

        if self.ot:
            noise.X, x_1, cond_args = self.batch_ot(noise.X, x_1, cond_args)
            noise.E, e_1, cond_args = self.batch_ot(noise.E, e_1, cond_args)


        noise.X, vf[0] = self.vecfield(self.preprocess(noise.X), self.preprocess(x_1), t[:, None], self.eps)
        noise.E, vf[1] = self.vecfield(self.preprocess(noise.E), self.preprocess(e_1), t[:, None], self.eps)

        # noise.X = noise.X.view(-1, *self.data_dims[0], self.n_class[0])
        # noise.E = noise.E.view(-1, *self.data_dims[1], self.n_class[1])

        upper_triangular_mask = torch.zeros_like(noise.E)
        indices = torch.triu_indices(row=noise.E.size(1), col=noise.E.size(2), offset=1)
        upper_triangular_mask[:, indices[0], indices[1], :] = 1

        noise.E = noise.E * upper_triangular_mask
        noise.E = noise.E + torch.transpose(noise.E, 1, 2)

        assert (noise.E == torch.transpose(noise.E, 1, 2)).all()

        noise = noise.mask(node_mask)


        pred_vf = self(t, noise, node_mask, *cond_args)


        loss = (self.norm2(noise.X, pred_vf[0].view(-1, *tuple(self.data_dims[0]), self.n_class[0]) - vf[0], self.eps).mean() 
                + 
                self.norm2(noise.E, pred_vf[1].view(-1, *tuple(self.data_dims[1]), self.n_class[1]) - vf[1], self.eps).mean())
 
        return loss/2

    #################################################
    # Sampling                                      #
    #################################################

    @torch.no_grad()
    def sample_euler(self, n_sample, n_steps, node_mask, device, *cond_args, return_traj=False):
        """
        Sampling using Euler method.
        :param n_sample: number of samples
        :param n_steps: number of Euler steps
        :param device: device to put the Tensor on
        :param cond_args: optional conditional arguments
        :param return_traj: whether to return the whole sampling trajectory
        :return: sampled data points of shape (n_sample, D, n) or (n_steps, n_sample, D, n)
        """
        p0 = self.preprocess(self.sample_prior(n_sample, self.total_data_dim, self.n_class, device=device))
        ps = [p0]
        p = p0
        ts = torch.linspace(0, 1, n_steps + 1, device=device)
        dt = 1 / n_steps
        for t in tqdm(ts[:-1]):
            pred_vf = self(torch.full((n_sample,), t, device=device), p, node_mask, *cond_args)
            p = self.exp(p, pred_vf * dt, self.eps)
            p = self.proj_x(p)
            if return_traj:
                ps.append(p)
        if return_traj:
            ps = torch.stack(ps, dim=0)
            return self.postprocess(ps)
        return self.postprocess(p)

    @torch.no_grad()
    def sample_ode(self, num_nodes, n_sample, n_steps, device, *cond_args, return_traj=False):
        """
        Sampling using ODE solver.
        :param num_nodes: number of nodes in the sampled graphs
        :param n_sample: number of samples
        :param n_steps: number of trajectory points
        :param device: device to put the Tensor on
        :param cond_args: optional conditional arguments
        :param return_traj: whether to return the whole sampling trajectory
        :return: sampled data points of shape (n_sample, D, n) or (n_steps, n_sample, D, n)
        """
        if not return_traj:
            n_steps = 1

        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(self.config["batch_size"], self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(
                self.config["batch_size"], device=self.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        arange = (
            torch.arange(torch.max(n_max).item(), device=self.device)
            .unsqueeze(0)
            .expand(self.config["batch_size"], -1)
        )
        node_mask = (arange < n_nodes.unsqueeze(1)).to(self.device)


        noise = PlaceHolder(
                            X=self.preprocess(self.sample_prior(n_sample, self.total_data_dim[0], self.n_class[0], device=device)), 
                            E=self.preprocess(self.sample_prior(n_sample, self.total_data_dim[1], self.n_class[1], device=device)), 
                            y=self.preprocess(self.sample_prior(n_sample, self.n_class[2], device=device))
                            )

        upper_triangular_mask = torch.zeros_like(noise.E)
        indices = torch.triu_indices(row=noise.E.size(1), col=noise.E.size(2), offset=1)
        upper_triangular_mask[:, indices[0], indices[1], :] = 1

        noise.E = noise.E * upper_triangular_mask
        noise.E = noise.E + torch.transpose(noise.E, 1, 2)

        assert (noise.E == torch.transpose(noise.E, 1, 2)).all()

        noise.mask(node_mask)

        # ps = odeint(
        #     lambda t, p: self(t, p, node_mask, *cond_args),
        #     p0,
        #     t=torch.linspace(0, 1, n_steps + 1, device=device, dtype=torch.float),
        #     atol=1e-5,
        #     rtol=1e-5,
        # )
        gen_X, gen_E = odeint(
            lambda t, p: self(t, p, node_mask, *cond_args),
            x=(noise.X, noise.E, noise.y),
            t0=0.0,
            t1=1.0,
            phi=self.parameters(),
        )

        # if return_traj:
        #     return self.postprocess(self.proj_x(ps))
        # return self.postprocess(self.proj_x(ps[-1]))
        return gen_X, gen_E

    def sample(self, method, num_nodes, n_sample, n_steps, device, *cond_args, return_traj=False):
        """
        Sample from the flow model.
        :param method: sampling method, should be 'euler' or 'ode'
        :param n_sample: number of samples
        :param n_steps: number of Euler steps or ODE trajectory points
        :param device: device to put the Tensor on
        :param cond_args: optional conditional arguments
        :param return_traj: whether to return the whole sampling trajectory
        :return: sampled data points of shape (n_sample, D, n) or (n_steps, n_sample, D, n)
        """
        assert method in ['euler', 'ode'], f'Unknown sampling method: {method}'
        # print(f'Start {method.upper()} sampling')
        sample_fn = self.sample_euler if method == 'euler' else self.sample_ode
        res = sample_fn(num_nodes, n_sample, n_steps, device, *cond_args, return_traj=return_traj)
        # print(f'End {method.upper()} sampling')
        return res

    #################################################
    # NLL Calculation                               #
    #################################################

    @staticmethod
    def preprocess_logp(p, eps=1e-4):
        """
        Calculate the log of the change of measure term introduced by the preprocess.
        :param p: data BEFORE preprocessing, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: log probability, Tensor of shape (...)
        """
        return torch.zeros(1, device=p.device, dtype=torch.float)

    @staticmethod
    def postprocess_logp(p, eps=1e-4):
        """
        Calculate the log of the change of measure term introduced by the postprocess.
        :param p: data AFTER postprocessing, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: log probability, Tensor of shape (...)
        """
        return torch.zeros(1, device=p.device, dtype=torch.float)

    def nll_forward(self, t, p_div):
        r"""
        Forward function for NLL calculation, using Hutchinson's trace estimator.

        .. math ::
            \text{div} f = \mathbb{E}_{\varepsilon\sim \mathcal{N}(0, I)}[\varepsilon^\top \nabla f \varepsilon]

        :param t: current timestep
        :param p_div: data point and divergence, Tensor of shape (B, D, n+1)
        :return: vector field and divergence, Tensor of shape (B, D, n+1)
        """
        p1 = p_div[..., :-1]
        vf, vjpfunc = vjp(lambda p: self(t, p), p1)
        v = self._v
        vJ = vjpfunc(v)[0]
        div = (vJ * v).sum(-1, keepdim=True)
        return torch.cat([vf, div], dim=-1)

    def nll_forward_exact(self, t, p_div):
        r"""
        Forward function for NLL calculation, exact calculation using autograd.

        .. math ::
            \text{div} f = \text{tr}(\nabla f)

        :param t: current timestep
        :param p_div: data point and divergence, Tensor of shape (B, 1, n+1)
        :return: vector field and divergence, Tensor of shape (B, 1, n+1)
        """
        assert p_div.size(1) == 1, 'exact divergence calculation only supports data dim of 1'

        def div_fn(u):
            r"""Accepts a function :math:`u: \mathbb{R}^D \to\mathbb{R}^D` and calculate the divergence."""
            J = jacrev(u)
            return lambda x: torch.trace(J(x))

        def vecfield(p):
            r"""Vector field :math:`u: \mathbb{R}^D \to\mathbb{R}^D`."""
            # unsqueeze dimensions to make the model happy
            return self(t, p.view(1, 1, -1)).view(-1)

        p1 = p_div[..., :-1]  # (B, 1, n)
        vf = self(t, p1)  # (B, 1, n)
        div = vmap(div_fn(vecfield))(p1.squeeze(1))  # (B,)
        return torch.cat([vf, div[:, None, None]], dim=-1)

    @torch.no_grad()
    def compute_nll_ode(self, p1, n_steps=200, tmax=1., tmin=0., exact=False, verbose=False):
        """
        Calculate the negative log-likelihood (NLL) using ODE solver.
        Points should not lie on the boundary of the simplex. If so use `compute_elbo_nll` instead.
        :param p1: data points, Tensor of shape (B, D, n)
        :param n_steps: not used
        :param tmax: last timestep
        :param tmin: first timestep
        :param exact: whether to use exact divergence calculation
        :param verbose: whether to print intermediate results
        :return: average NLL
        """
        device = p1.device
        self._v = torch.randn_like(p1)
        nll_fn = self.nll_forward_exact if exact else self.nll_forward
        init_div = torch.zeros(p1.size(0), self.total_data_dim, 1, device=device, dtype=torch.float)

        state1 = torch.cat([self.preprocess(p1), init_div], dim=-1)
        state0 = odeint(
            nll_fn,
            state1,
            t=torch.linspace(tmax, tmin, 2, device=device, dtype=torch.float),
            atol=1e-4,
            rtol=1e-4,
        )[-1]

        p0 = self.postprocess(self.proj_x(state0[..., :-1]))
        logdetjac = state0[..., -1]
        logp0 = self.prior_logp0(p0, self.eps)
        logt0 = self.preprocess_logp(p0, self.eps)
        logt1 = self.postprocess_logp(p1, self.eps)
        if verbose:
            print(
                f'logp0: {logp0.mean().item():.4f}, '
                f'logdetjac: {logdetjac.mean().item():.4f}, '
                f'logt0: {logt0.mean().item():.4f}, '
                f'logt1: {logt1.mean().item():.4f}'
            )

        logp1 = logp0 + logdetjac + logt0 + logt1
        return -logp1.mean()

    @torch.no_grad()
    def compute_nll_euler(self, p1, n_steps=200, tmax=1., tmin=0., exact=False, verbose=False):
        """
        Calculate the negative log-likelihood (NLL) using Euler method.
        Points should not lie on the boundary of the simplex. If so use `computer_elbo_euler` instead.
        :param p1: data points, Tensor of shape (B, D, n)
        :param n_steps: number of Euler steps
        :param tmax: last timestep
        :param tmin: first timestep
        :param exact: whether to use exact divergence calculation
        :param verbose: whether to print intermediate results
        :return: average NLL
        """
        device = p1.device
        self._v = torch.randn_like(p1)
        nll_fn = self.nll_forward_exact if exact else self.nll_forward
        logdetjac = torch.zeros(p1.size(0), self.total_data_dim, 1, device=device, dtype=torch.float)

        ts = torch.linspace(tmax, tmin, n_steps, device=device, dtype=torch.float)
        dt = 1 / n_steps
        p_hat = self.preprocess(p1)
        for t in tqdm(ts, desc='Euler NLL'):
            vf_div = nll_fn(t, torch.cat([p_hat, logdetjac], dim=-1))
            vf, div = vf_div[..., :-1], vf_div[..., -1:]
            p_hat = self.exp(p_hat, -vf * dt, self.eps)
            p_hat = self.proj_x(p_hat)
            logdetjac = logdetjac - div * dt

        p0 = self.postprocess(p_hat)
        logp0 = self.prior_logp0(p0, self.eps)
        logdetjac = logdetjac.squeeze(-1)
        logt0 = self.preprocess_logp(p0, self.eps)
        logt1 = self.postprocess_logp(p1, self.eps)
        if verbose:
            print(
                f'logp0: {logp0.mean().item():.4f}, '
                f'logdetjac: {logdetjac.mean().item():.4f}, '
                f'logt0: {logt0.mean().item():.4f}, '
                f'logt1: {logt1.mean().item():.4f}'
            )

        logp1 = logp0 + logdetjac + logt0 + logt1
        return -logp1.mean()

    def compute_nll(self, method, p1, n_steps=200, tmax=1., tmin=0., exact=False, verbose=False):
        """
        Calculate the negative log-likelihood (NLL).
        Points should not lie on the boundary of the simplex. If so use `computer_elbo` instead.
        :param method: sampling method, should be 'euler' or 'ode'
        :param p1: data points, Tensor of shape (B, D, n)
        :param n_steps: number of Euler steps or ODE trajectory points
        :param tmax: last timestep
        :param tmin: first timestep
        :param exact: whether to use exact divergence calculation
        :param verbose: whether to print intermediate results
        :return: average NLL
        """
        assert method in ['euler', 'ode'], f'Unknown NLL method: {method}'
        nll_fn = self.compute_nll_euler if method == 'euler' else self.compute_nll_ode
        nll = nll_fn(p1, n_steps, tmax=tmax, tmin=tmin, exact=exact, verbose=verbose)
        return nll

    @torch.no_grad()
    def compute_elbo_ode(self, p1, n_steps=200, tmax=0.995, verbose=False):
        """
        Compute the evidence lower bound (ELBO) for NLL using ODE solver.
        :param p1: data points, Tensor of shape (B, D, n)
        :param n_steps: not used
        :param tmax: last timestep, should be close to but less than 1
        :param verbose: whether to print intermediate results
        :return: average ELBO
        """
        p1_bar, logq = self.sample_simplex_linear(p1, tmax)
        nll = self.compute_nll_ode(p1_bar, n_steps=n_steps, tmax=tmax, verbose=verbose)
        logx = (torch.log(p1_bar.clamp(min=1e-4)) * (p1 > 0.5)).sum(-1)
        logp1 = -nll + logx - logq
        if verbose:
            print(
                f'nll: {nll.mean().item():.4f}, '
                f'logx: {logx.mean().item():.4f}, '
                f'logq: {logq.mean().item():.4f}'
            )
        return -logp1.mean()

    @torch.no_grad()
    def compute_elbo_euler(self, p1, n_steps=200, tmax=0.995, verbose=False):
        """
        Compute the evidence lower bound (ELBO) for NLL using Euler method.
        :param p1: data points, Tensor of shape (B, D, n)
        :param n_steps: number of Euler steps
        :param tmax: last timestep, should be close to but less than 1
        :param verbose: whether to print intermediate results
        :return: average ELBO
        """
        p1_bar, logq = self.sample_simplex_linear(p1, tmax)
        nll = self.compute_nll_euler(p1_bar, n_steps=n_steps, tmax=tmax, verbose=verbose)
        logx = (torch.log(p1_bar.clamp(min=1e-4)) * (p1 > 0.5)).sum(-1)
        logp1 = -nll + logx - logq
        if verbose:
            print(
                f'nll: {nll.mean().item():.4f}, '
                f'logx: {logx.mean().item():.4f}, '
                f'logq: {logq.mean().item():.4f}'
            )
        return -logp1.mean()

    def compute_elbo(self, method, p1, n_steps=200, tmax=0.995, verbose=False):
        """
        Compute the evidence lower bound (ELBO) for NLL.
        :param method: sampling method, should be 'euler' or 'ode'
        :param p1: data points, Tensor of shape (B, D, n)
        :param n_steps: number of Euler steps or ODE trajectory points
        :param tmax: last timestep, should be close to but less than 1
        :param verbose: whether to print intermediate results
        :return: average ELBO
        """
        assert method in ['euler', 'ode'], f'Unknown ELBO method: {method}'
        elbo_fn = self.compute_elbo_euler if method == 'euler' else self.compute_elbo_ode
        elbo = elbo_fn(p1, n_steps, tmax=tmax, verbose=verbose)
        return elbo
    






class GraphStatFlow_Simplex(GraphStatFlow):
    r"""
    Naive implementation of the Fisher metric induced Riemannian geometry on the probability simplex.
    The metric is not defined at the boundary of the simplex.
    In order to be comparable to LinearFM, the divergence calculated here is also Euclidean.
    """

    @staticmethod
    def prior_logp0(p0, eps=1e-4):
        n_class = torch.tensor(p0.size(-1), dtype=torch.float)
        return torch.lgamma(n_class)

    @classmethod
    def proj_x(cls, x, eps=0.):
        x = x.clamp(eps, 1 - eps)
        return x / x.sum(dim=-1, keepdim=True)

    @classmethod
    def proj_vf(cls, vf, pt):
        return vf - vf.mean(dim=-1, keepdim=True)

    @classmethod
    def dist(cls, p, q, eps=1e-2):
        return 2 * torch.acos((p * q).sqrt().sum(-1).clamp(0, 1 - eps))

    @classmethod
    def norm2(cls, p, u, eps=1e-2):
        mask = (p > eps).float()
        return (u.square() / p.clamp(min=eps) * mask).sum(-1)

    @classmethod
    def exp(cls, p, u, eps=1e-2):
        s = p.sqrt()
        xs = u / s.clamp(min=eps) / 2
        theta = torch.norm(xs, dim=-1, keepdim=True)
        return (s * torch.cos(theta) + xs * torch.sinc(theta / torch.pi)) ** 2

    @classmethod
    def log(cls, p, q, eps=1e-2):
        z = torch.sqrt(p * q)
        s = z.sum(-1, keepdim=True)
        dist = 2 * torch.acos(s.clamp(0, 1 - eps))
        u = dist / torch.sqrt((1 - s ** 2).clamp(min=eps)) * (z - s * p)
        return torch.where(dist > eps, u, q - p)

    @classmethod
    def vecfield(cls, p, q, t, eps=1e-2):
        t_reshaped = t.view(-1, *([1]*(len(p.size())-1)))
        pt = cls.interpolate(p, q, t_reshaped, eps)
        vf = cls.log(pt, q, eps) / (1 - t_reshaped)
        return pt, vf
    

class GraphStatFlow_Sphere(GraphStatFlow):
    r"""
    Implementation of the Fisher metric induced Riemannian geometry on the probability simplex
    by leverage the diffeomorphism between the probability simplex and the sphere.
    The Riemannian metric, exponential, and logarithm maps can be extended to the boundary of the simplex
    The mapping is an isometry up to a constant factor of 2, and the flow is more stable to train.

    .. math ::
        \pi: \Delta^{n-1} \to S^{n-1}, \quad p \mapsto \sqrt{p}
    """

    @staticmethod
    def prior_logp0(p0, eps=1e-4):
        n_class = torch.tensor(p0.size(-1), dtype=torch.float)
        return torch.lgamma(n_class)

    @staticmethod
    def preprocess(p):
        return p.sqrt()

    @staticmethod
    def postprocess(p_hat):
        return p_hat.square()

    @classmethod
    def proj_x(cls, x, eps=0.):
        x = x.clamp(eps, 1 - eps)
        return F.normalize(x, dim=-1)

    @classmethod
    def proj_vf(cls, vf, pt):
        return vf - (pt * vf).sum(dim=-1, keepdim=True) * pt

    @classmethod
    def dist(cls, p, q, eps=1e-4):
        return torch.acos((p * q).sum(-1).clamp(0, 1 - eps))

    @classmethod
    def norm2(cls, p, u, eps=1e-4):
        return u.square().sum(-1)

    @classmethod
    def exp(cls, p, u, eps=1e-4):
        u_norm = u.norm(dim=-1, keepdim=True)
        exp = torch.cos(u_norm) * p + torch.sinc(u_norm / torch.pi) * u
        return exp

    @classmethod
    def log(cls, p, q, eps=1e-4):
        x, y = p, q
        u = y - x
        u = u - (x * u).sum(dim=-1, keepdim=True) * x
        u_norm = u.norm(dim=-1, keepdim=True).clamp(min=eps)
        dist = cls.dist(x, y, eps).unsqueeze(-1)
        return torch.where(dist > eps, u * dist / u_norm, u)

    @classmethod
    def vecfield(cls, p, q, t, eps=1e-4):
        x, y = p, q
        dist = cls.dist(x, y, eps).unsqueeze(-1)
        xt = cls.interpolate(x, y, t, eps)
        ux = x - xt
        ux = ux - (xt * ux).sum(dim=-1, keepdim=True) * xt
        ux_norm = ux.norm(dim=-1, keepdim=True)
        uy = y - xt
        uy = uy - (xt * uy).sum(dim=-1, keepdim=True) * xt
        vf = dist * torch.where(ux_norm > eps, -ux / ux_norm, F.normalize(uy, dim=-1))
        return xt, vf

    @staticmethod
    def preprocess_logp(p, eps=1e-4):
        n_class = torch.tensor(p.size(-1), dtype=torch.float, device=p.device)
        return torch.log(p.sqrt().clamp(min=eps)).sum(-1) + np.log(2) * (n_class - 1)

    @staticmethod
    def postprocess_logp(p, eps=1e-4):
        n_class = torch.tensor(p.size(-1), dtype=torch.float, device=p.device)
        return -torch.log(p.sqrt().clamp(min=eps)).sum(-1) - np.log(2) * (n_class - 1)


class GraphStatFlow_Linear(GraphStatFlow):
    """
    Simple linear flow model on the probability simplex. Assume an underlying Euclidean geometry.
    """

    @staticmethod
    def prior_logp0(p0, eps=1e-4):
        n_class = torch.tensor(p0.size(-1), dtype=torch.float)
        return torch.lgamma(n_class)

    @classmethod
    def proj_x(cls, x, eps=0.):
        x = x.clamp(eps, 1 - eps)
        return x / x.sum(dim=-1, keepdim=True)

    @classmethod
    def proj_vf(cls, vf, pt):
        return vf - vf.mean(dim=-1, keepdim=True)

    @classmethod
    def dist(cls, p, q, eps=1e-4):
        return torch.norm(p - q, p=2, dim=-1)

    @classmethod
    def norm2(cls, p, u, eps=1e-4):
        return u.square().sum(-1)

    @classmethod
    def exp(cls, p, u, eps=1e-4):
        return p + u

    @classmethod
    def log(cls, p, q, eps=1e-4):
        return q - p

    @classmethod
    def vecfield(cls, p, q, t, eps=1e-4):
        t_expand = t.unsqueeze(-1)
        pt = p * (1 - t_expand) + q * t_expand
        return pt, q - p
