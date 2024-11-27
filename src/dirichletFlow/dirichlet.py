import torch
import torch.nn.functional as F
import einops
from torch import nn
from zuko.utils import odeint
from src.catflow.utils import timestep_embedding
from src.catflow.transformer_model import GraphTransformer
from src.dirichletFlow.flow_utils import GaussianFourierProjection

import src.catflow.utils as utils
from src.qm9 import qm9_dataset
from src.qm9.extra_features_molecular import ExtraMolecularFeatures

class DirichletFlow(nn.Module):
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
            backbone_model (Backbone): Backbone model to extract features. In the case of our experiments, we use a graph transformer network.
            batch_size (int): Batch size. Default value is 32.
            num_nodes (int): Number of nodes. Default value is 10.
            num_classes (int): Number of classes. Default value is 10.
            eps (float): Epsilon value to avoid numerical instability. Default value is 1e-6.
        """
        super(DirichletFlow, self).__init__()
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
        self.time_embedder_node = nn.Sequential(GaussianFourierProjection(embedding_dim=config["n_node_classes"]+2), nn.Linear(config["n_node_classes"]+2, config["n_node_classes"]+2))
        self.time_embedder_edge = nn.Sequential(GaussianFourierProjection(embedding_dim=config["n_edge_classes"]), nn.Linear(config["n_edge_classes"], config["n_edge_classes"]))
        # self.cls_head = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim),
        #                 nn.ReLU(),
        #                 nn.Linear(args.hidden_dim, self.num_cls))

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
            x (torch.tensor): Node noise. Shape: (batch_size, num_nodes, num_classes_nodes).
            e (torch.tensor): Edge noise. Shape: (batch_size, num_nodes, num_nodes, num_classes_edges).


        Returns:
            torch.tensor: Parameters of the variational distribution. Shape: (batch_size, num_nodes + (num_nodes - 1)^2, num_classes + 1).
        """
        x = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        e = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        # embed the timestep using GuassianProjection
        time_emb_n = F.relu(self.time_embedder_node(t))
        time_emb_e = F.relu(self.time_embedder_edge(t))
        x += einops.rearrange(time_emb_n, "b c -> b 1 c")
        e += einops.rearrange(time_emb_e, "b c -> b 1 1 c")
        # x = x + time_emb_n[:,None,:]
        # e = e + time_emb_e[:,None,:]

        #return self.cls_head(feat)

        return self.backbone_model(
            X=x,
            E=e,
            # dummy y
            y=y,
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
