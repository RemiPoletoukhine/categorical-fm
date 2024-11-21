import torch
import einops
from torch import nn
from functools import partial
from zuko.utils import odeint
from src.qm9 import qm9_dataset
import src.catflow.utils as utils
from src.catflow.transformer_model import GraphTransformer
from src.qm9.extra_features_molecular import ExtraMolecularFeatures


class CatFlow(nn.Module):
    def __init__(
        self,
        config: dict,
        dataset_infos: qm9_dataset.QM9infos,
        domain_features: ExtraMolecularFeatures,
        device: torch.device,
        eps: float = 1e-5,
    ) -> None:
        """
        Constructor of the CatFlow model.

        Args:
            config (dict): Configuration of the model.
            dataset_infos (QM9infos): Information about the dataset.
            domain_features (ExtraMolecularFeatures): Extra molecular features to append to the input.
            device (torch.device): Device to run the model.
            eps (float): Epsilon value to avoid numerical instability. Default value is 1e-5.
        """
        super(CatFlow, self).__init__()
        self.config = config
        self.dataset_info = dataset_infos
        self.domain_features = domain_features
        self.node_dist = self.dataset_info.nodes_dist
        self.device = device
        self.backbone_model = GraphTransformer(
            n_layers=config["n_layers"],
            input_dims=self.dataset_info.input_dims,
            hidden_mlp_dims=config["hidden_mlp_dims"],
            hidden_dims=config["hidden_dims"],
            output_dims=self.dataset_info.output_dims,
        ).to(self.device)
        self.num_classes_nodes = config["num_classes"]["X"]
        self.num_classes_edges = config["num_classes"]["E"]
        self.eps = eps

    def sample_time(self, batch_size) -> torch.tensor:
        """
        Function to sample the time step for the CatFlow model.

        Args:
            batch_size (int): Batch size.

        Returns:
            torch.tensor: Time step. Shape: (batch_size,).
        """
        # We sample the time from U[0, 1 - eps] as in Flow Matching for Generative Modeling by Yaron Lipman et al.
        # source: https://github.com/gle-bellier/flow-matching
        return (
            (torch.rand(1) + torch.arange(batch_size) / batch_size) % (1 - self.eps)
        ).to(self.device)

    def sample_noise(self, x, e, y, node_mask) -> utils.PlaceHolder:
        """
        Function to sample the noise for the CatFlow model.

        Args:
            x (torch.tensor): Node features, only for the shape. Shape: (batch_size, num_nodes, num_node_classes).
            e (torch.tensor): Edge features, only for the shape. Shape: (batch_size, num_nodes, num_nodes, num_edge_classes).
            y (torch.tensor): Global features, only for the shape. Shape: (batch_size, y_emb).
            node_mask (torch.tensor): Node mask. Shape: (batch_size, num_nodes).

        Returns:
            utils.PlaceHolder: Extra data to append to the network input. Keys: 'X', 'E', 'y'. Values: torch.tensor.

        """
        # Judging by the page 7 of the paper: the noise is not constrained to the simplex; so we can sample from a normal distribution
        sampled = utils.sample_normal(
            torch.zeros_like(x),
            torch.zeros_like(e),
            torch.zeros_like(y),
            torch.tensor([1]).unsqueeze(0).to(self.device),
            node_mask,
        )
        return sampled

    def compute_extra_data(self, noisy_data: dict) -> utils.PlaceHolder:
        """
        Function to compute extra information and append to the network input at every training step (after adding noise)

        Args:
            noisy_data (dict): Noisy data. Keys: 'X_t', 'E_t', 'y_t', 't', 'node_mask'. Values: torch.tensor.

        Returns:
            utils.PlaceHolder: Extra data to append to the network input. Keys: 'X', 'E', 'y'. Values: torch.tensor.
        """

        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = extra_molecular_features.X
        extra_E = extra_molecular_features.E
        extra_y = extra_molecular_features.y

        t = noisy_data["t"].unsqueeze(-1)
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def forward(
        self,
        t: torch.tensor,
        noisy_data: dict,
        extra_data: utils.PlaceHolder,
        node_mask: torch.tensor,
    ) -> utils.PlaceHolder:
        """
        Forward pass of the backbone model.

        Args:
            t (torch.tensor): Time step. Shape: (batch_size,).
            noisy_data (dict): Noisy data. Keys: 'X_t', 'E_t', 'y_t'. Values: torch.tensor.
            extra_data (utils.PlaceHolder): Extra data to append to the network input. Keys: 'X', 'E', 'y'. Values: torch.tensor.
            node_mask (torch.tensor): Node mask. Shape: (batch_size, num_nodes).

        Returns:
            utils.PlaceHolder: Inferred state. Keys: 'X', 'E', 'y'. Values: torch.tensor.
        """
        # add the extra data to the input
        x = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        e = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()

        # embed the timestep using the sinusoidal positional encoding
        t_embedded_nodes = utils.timestep_embedding(
            t, dim=x.shape[-1]
        )  # Shape: (batch_size, num_classes_nodes)
        t_embedded_edges = utils.timestep_embedding(
            t, dim=e.shape[-1]
        )  # Shape: (batch_size, num_classes_edges)

        # add time embedding to the input across the class feature dimension
        x += einops.rearrange(t_embedded_nodes, "b c -> b 1 c")
        e += einops.rearrange(t_embedded_edges, "b c -> b 1 1 c")

        return self.backbone_model(
            X=x,
            E=e,
            y=y,
            node_mask=node_mask,
        )

    def vector_field(
        self,
        t: float,
        X_0,
        E_0,
        y_0,
        node_mask: torch.tensor,
    ) -> torch.tensor:
        """
        Function that returns the vector field of the CatFlow model for a given timestamp.

        Args:
            t (float): Time step.
            X_0 (torch.tensor): Initial node features. Shape: (batch_size, num_nodes, num_node_classes).
            E_0 (torch.tensor): Initial edge features. Shape: (batch_size, num_nodes, num_nodes, num_edge_classes).
            y_0 (torch.tensor): Initial global features. Shape: (batch_size, y_emb).
            node_mask (torch.tensor): Node mask. Shape: (batch_size, num_nodes).

        Returns:
            torch.tensor: Vector field. Shape: (batch_size, num_nodes + (num_nodes - 1)**2, num_classes + 1).
        """
        print(f"t: {t}")
        t_expanded = t.expand(X_0.shape[0])
        noisy_data = {
            "X_t": X_0,
            "E_t": E_0,
            "y_t": y_0,
            "t": t_expanded,
            "node_mask": node_mask,
        }
        extra_data = self.compute_extra_data(noisy_data=noisy_data)

        # prediction of the vector field at time t: forward pass of the backbone model
        inferred_state = self.forward(t_expanded, noisy_data, extra_data, node_mask)

        node_repr = inferred_state.X
        edge_repr = inferred_state.E
        y_repr = inferred_state.y

        node_vector_field = (node_repr - noisy_data["X_t"]) / (1 - t + self.eps)
        edge_vector_field = (edge_repr - noisy_data["E_t"]) / (1 - t + self.eps)
        y_vector_field = (y_repr - noisy_data["y_t"]) / (1 - t + self.eps)

        return node_vector_field, edge_vector_field, y_vector_field

    def sampling(
        self,
        t_0: float = 0.0,
        t_1: float = 1.0,
        num_nodes: int = None,
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        Function to sample a new batch of graphs following the learned vector field.

        Args:
            t_0 (float): Initial time step. Default value is 0.0.
            t_1 (float): Final time step. Default value is 1.0.
            num_nodes (int): Number of nodes for the sampled graphs. Default value is None.

        Returns:
            tuple[torch.tensor, torch.tensor]: Final node and edge representations. Shapes: (batch_size, num_nodes, num_node_classes), (batch_size, num_nodes, num_nodes, num_edge_classes).
        """
        # Optionally: set the same number of nodes for all the generated graphs
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
        # Build the masks
        arange = (
            torch.arange(n_max, device=self.device)
            .unsqueeze(0)
            .expand(self.config["batch_size"], -1)
        )
        node_mask = (arange < n_nodes.unsqueeze(1)).to(self.device)
        # Sample noise x_0, e_0, y_0
        sampled = utils.sample_normal(
            mu_X=torch.zeros(
                (self.config["batch_size"], n_max, self.num_classes_nodes)
            ).to(self.device),
            mu_E=torch.zeros(
                (self.config["batch_size"], n_max, n_max, self.num_classes_edges)
            ).to(self.device),
            mu_y=torch.zeros((self.config["batch_size"], 0)).to(self.device),
            sigma=torch.tensor([1]).unsqueeze(0).to(self.device),
            node_mask=node_mask,
        )
        x_0, e_0, y_0 = sampled.X, sampled.E, sampled.y
        # Run the ODE solver to perform the integration of the vector field
        final_state = odeint(
            f=partial(self.vector_field, node_mask=node_mask),
            x=(x_0, e_0, y_0),
            t0=t_0,
            t1=t_1,
            phi=self.parameters(),
        )
        # Separate the final node and edge representations
        (
            final_nodes,
            final_edges,
            _,
        ) = final_state

        return final_nodes, final_edges
