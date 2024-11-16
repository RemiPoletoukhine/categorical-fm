import torch
import einops
from torch import nn
from zuko.utils import odeint
from src.catflow.utils import timestep_embedding
from src.catflow.transformer_model import GraphTransformer


class CatFlow(nn.Module):
    def __init__(
        self,
        config: dict,
        device: torch.device,
        eps: float = 1e-6,
    ) -> None:
        """
        Constructor of the CatFlow model.

        Args:
            backbone_model (Backbone): Backbone model to extract features. In the case of our experiments, we use a graph transformer network.
            batch_size (int): Batch size. Default value is 32.
            num_nodes (int): Number of nodes. Default value is 10.
            num_classes (int): Number of classes. Default value is 10.
            eps (float): Epsilon value to avoid numerical instability. Default value is 1e-6.
        """
        super(CatFlow, self).__init__()
        self.device = device
        self.backbone_model = GraphTransformer(
            n_layers=config["n_layers"],
            input_dims=config["input_dims"],
            hidden_mlp_dims=config["hidden_mlp_dims"],
            hidden_dims=config["hidden_dims"],
            output_dims=config["output_dims"],
        ).to(self.device)
        self.batch_size = config["batch_size"]
        self.num_classes_nodes = config["input_dims"]["X"]
        self.num_classes_edges = config["input_dims"]["E"]
        self.eps = eps

    def sample_time(self, lambd: torch.tensor = torch.tensor([1.0])) -> torch.tensor:
        """
        Function to sample the time step for the CatFlow model.

        Args:
            lambd (torch.tensor): Rate parameter of the exponential distribution. Default value is 1.0.

        Returns:
            torch.tensor: Time step. Shape: (batch_size,).
        """
        # As in Dirichlet Flow Matching, we sample the time step from Exp(1)
        return (
            torch.distributions.exponential.Exponential(lambd)
            .sample()
            .expand(self.batch_size)
        )

    def sample_noise(self, kind, num_nodes) -> torch.tensor:
        """
        Function to sample the noise for the CatFlow model.

        Args:
            kind (str): Type of noise to sample. Options: 'node' or 'edge'.

        Returns:
            torch.tensor: Noise. Shape: (batch_size, num_nodes + (num_nodes - 1)**2, num_classes + 1).
        """
        # Judging by the page 7 of the paper: the noise is not constrained to the simplex; so we can sample from a normal distribution
        if kind == "node":
            return torch.randn(self.batch_size, num_nodes, self.num_classes_nodes)
        elif kind == "edge":
            return torch.randn(
                self.batch_size, num_nodes, num_nodes, self.num_classes_edges
            )
        else:
            raise ValueError(f"Invalid noise type: {kind}")

    def forward(
        self, t: torch.tensor, x: torch.tensor, e: torch.tensor, node_mask: torch.tensor
    ) -> torch.tensor:
        """
        Forward pass of the backbone model.

        Args:
            t (torch.tensor): Time step. Shape: (batch_size,).
            x (torch.tensor): Node noise. Shape: (batch_size, num_nodes, num_classes_nodes).
            e (torch.tensor): Edge noise. Shape: (batch_size, num_nodes, num_nodes, num_classes_edges).


        Returns:
            torch.tensor: Parameters of the variational distribution. Shape: (batch_size, num_nodes + (num_nodes - 1)^2, num_classes + 1).
        """
        # embed the timestep using the sinusoidal positional encoding
        t_embedded_nodes = timestep_embedding(
            t, dim=x.shape[-1]
        )  # Shape: (batch_size, num_classes_nodes)
        t_embedded_edges = timestep_embedding(
            t, dim=e.shape[-1]
        )  # Shape: (batch_size, num_classes_edges)
        # add time embedding to the input across the class feature dimension
        x += einops.rearrange(t_embedded_nodes, "b c -> b 1 c")
        e += einops.rearrange(t_embedded_edges, "b c -> b 1 1 c")

        return self.backbone_model(
            X=x,
            E=e,
            # dummy y
            y=torch.ones(self.batch_size, 1).to(self.device),
            node_mask=node_mask,
        )

    def vector_field(
        self, t: torch.tensor, x: torch.tensor, e: torch.tensor, node_mask: torch.tensor
    ) -> torch.tensor:
        """
        Function that returns the vector field of the CatFlow model for a given timestamp.

        Args:
            t (torch.tensor): Time step. Shape: (batch_size, 1).
            x (torch.tensor): Input noise. Shape: (batch_size, num_nodes + (num_nodes - 1)**2, num_classes + 1).

        Returns:
            torch.tensor: Vector field. Shape: (batch_size, num_nodes + (num_nodes - 1)**2, num_classes + 1).
        """
        # repeat the time step to match the shape of the input noise
        t_expanded = t.expand(self.batch_size)
        # prediction of the vector field at time t: forward pass of the backbone model
        inferred_state = self.forward(t_expanded, x, e, node_mask)
        node_repr = inferred_state.X
        edge_repr = inferred_state.E

        node_vector_field = (node_repr - x) / (1 - t)
        edge_vector_field = (edge_repr - e) / (1 - t)

        return node_vector_field, edge_vector_field, node_mask

    def sampling(
        self,
        init_state: tuple[torch.tensor, torch.tensor, torch.tensor],
        t_0: float = 0.0,
        t_1: float = 0.95,
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        Function to sample a new instance following the learned vector field.

        Args:
            x (torch.tensor): Input node noise. Shape: (batch_size, num_nodes, num_classes + 1).
            e (torch.tensor): Input edge noise. Shape: (batch_size, num_nodes, num_nodes, num_classes + 1).

        Returns:
            torch.tensor: Sampled result.
        """
        # Run the ODE solver to perform the integration of the vector field
        final_state = odeint(
            f=self.vector_field, x=init_state, t0=t_0, t1=t_1, phi=self.parameters()
        )
        # Separate the final node and edge representations
        final_nodes, final_edges, _ = final_state

        return final_nodes, final_edges
