import torch
import torch.nn.functional as F
import einops
from torch import nn
from zuko.utils import odeint
from src.catflow.utils import timestep_embedding
from src.catflow.transformer_model import GraphTransformer
from src.dirichletFlow.flow_utils import GaussianFourierProjection,expand_simplex,simplex_proj,DirichletConditionalFlow

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
        self.condflow_node = DirichletConditionalFlow(K=config["n_node_classes"], alpha_spacing=0.001, alpha_max=config["alpha_max"])
        self.condflow_edge = DirichletConditionalFlow(K=config["n_edge_classes"], alpha_spacing=0.001, alpha_max=config["alpha_max"])
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
        self,
        t: float,
        X_0: torch.tensor,
        E_0: torch.tensor,
        y_0: torch.tensor,
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
        # track progress of the integration. Unfortunately, that's the prettiest way (of the simpler ones) to do it
        print(f"Current step: {t}")
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
        inferred_state = self.forward(noisy_data, extra_data, node_mask)

        node_repr = inferred_state.X
        edge_repr = inferred_state.E
        y_repr = inferred_state.y

        node_vector_field = (node_repr - noisy_data["X_t"]) / (1 - t + self.eps)
        edge_vector_field = (edge_repr - noisy_data["E_t"]) / (1 - t + self.eps)
        y_vector_field = (y_repr - noisy_data["y_t"]) / (1 - t + self.eps)

        return node_vector_field, edge_vector_field, y_vector_field

    def flow_inference(self,B,num_nodes,t,s,xt,flow_probs,eye,condflow):
        if not torch.allclose(flow_probs.sum(2), torch.ones((B, num_nodes), device=self.device), atol=1e-4) or not (flow_probs >= 0).all():
                print(f'WARNING: flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the simplex. There are we are {(flow_probs<0).sum()} negative values in flow_probs of shape {flow_probs.shape} that are negative. We are projecting them onto the simplex.')
                flow_probs = simplex_proj(flow_probs)

        c_factor = condflow.c_factor(xt.cpu().numpy(), s.item())
        c_factor = torch.from_numpy(c_factor).to(xt)

        #self.inf_counter += 1  TODO
        if torch.isnan(c_factor).any():
            print(f'NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')
            if self.args.allow_nan_cfactor:
                c_factor = torch.nan_to_num(c_factor)
                self.nan_inf_counter += 1
            else:
                raise RuntimeError(f'NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')

        if not (flow_probs >= 0).all(): print(f'flow_probs.min(): {flow_probs.min()}')
        cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
        flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)

        xt = xt + flow * (t - s)

        if not torch.allclose(xt.sum(2), torch.ones((B, num_nodes), device=self.device), atol=1e-4) or not (xt >= 0).all():
            print(f'WARNING: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative. We are projecting them onto the simplex.')
            xt = simplex_proj(xt)
        return xt
    
    @torch.no_grad()
    def sampling(
        self,
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
        node_mask = arange < n_nodes.unsqueeze(1)

        num_node_cls = self.config["n_node_classes"]
        num_edge_cls = self.config["n_edge_classes"]
        B = self.config["batch_size"]
        x_node = torch.distributions.Dirichlet(torch.ones(B, num_nodes, num_node_cls, device=self.device)).sample()
        eye_node = torch.eye(num_node_cls).to(x_node)
        xt_node = x_node.clone()

        x_edge = torch.distributions.Dirichlet(torch.ones(B, num_nodes, num_nodes, num_edge_cls, device=self.device)).sample()
        eye_edge = torch.eye(num_edge_cls).to(x_node)
        xt_edge = x_edge.clone()
        y = torch.empty((B, 0)).to(xt_node.device)
        # logits_pred, _ = self.dirichlet_flow_inference(seq, cls, model=self.model, args=self.args)
        # seq_pred = torch.argmax(logits_pred, dim=-1)
        
        t_span = torch.linspace(1, self.config["alpha_max"], self.config["num_integration_steps"], device=self.device)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            #xt_node_expanded, prior_weights = expand_simplex(xt_node, s[None].expand(B))
            #xt_edge_expanded, prior_weights = expand_simplex(xt_edge.reshape(B,num_nodes*num_nodes,num_edge_cls), s[None].expand(B))
            #_,_,new_cls = xt_edge_expanded.shape
            #xt_edge_expanded = xt_edge_expanded.reshape(B,num_nodes,num_nodes,new_cls)
            t_expanded = t.expand(xt_node.shape[0])
            noisy_data = {
            "X_t": xt_node,
            "E_t": xt_edge,
            "y_t": y,
            "t": t_expanded,
            "node_mask": node_mask,
            }
            extra_data = self.compute_extra_data(noisy_data=noisy_data)

            # prediction of the vector field at time t: forward pass of the backbone model
            inferred_state = self.forward(t_expanded,noisy_data, extra_data, node_mask)
            masked_pred_X=inferred_state.X
            masked_pred_E=inferred_state.E
            pred_y=inferred_state.y

            masked_pred_E = masked_pred_E.reshape(B,num_nodes*num_nodes,num_edge_cls)

            flow_probs_node = torch.nn.functional.softmax(masked_pred_X, -1) # [B, L, K]
            flow_probs_edge = torch.nn.functional.softmax(masked_pred_E, -1) # 

            xt_node = self.flow_inference(B,num_nodes,t,s,xt_node,flow_probs_node,eye_node,self.condflow_node)
            xt_edge = self.flow_inference(B,num_nodes*num_nodes,t,s,xt_edge.reshape(B,num_nodes*num_nodes,num_edge_cls),flow_probs_edge,eye_edge,self.condflow_edge)
            xt_edge = xt_edge.reshape(B,num_nodes,num_nodes,num_edge_cls)
        return inferred_state.X, inferred_state.E