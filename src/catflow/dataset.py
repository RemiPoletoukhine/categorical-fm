from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
import torch


class QM9GraphDataset(Dataset):
    def __init__(self, data, nodes_classes=5, edge_classes=4):
        self.data = data
        self.nodes_classes = nodes_classes
        self.edge_classes = edge_classes

    def __len__(self):
        return len(self.data["node_idx_array"])  # Number of graphs

    def __getitem__(self, idx):
        # Get node and edge ranges for the idx-th molecule
        node_start, node_end = self.data["node_idx_array"][idx]
        edge_start, edge_end = self.data["edge_idx_array"][idx]

        # Extract node and edge information
        atom_types = self.data["atom_types"][
            node_start:node_end
        ]  # Shape: [num_nodes, nodes_classes]
        bond_idxs = self.data["bond_idxs"][edge_start:edge_end]  # Shape: [num_edges, 2]
        bond_types = self.data["bond_types"][edge_start:edge_end].to(
            torch.long
        )  # Shape: [num_edges]

        num_nodes = atom_types.shape[0]

        # Initialize the node feature tensor (one-hot encoded node types)
        node_labels = torch.argmax(
            atom_types.to(torch.float), dim=1
        )  # Get the node class labels
        node_features = F.one_hot(
            node_labels, num_classes=self.nodes_classes
        )  # Shape: [num_nodes, nodes_classes]

        # Initialize edge feature tensor
        edge_class_matrix = torch.full(
            (num_nodes, num_nodes), self.edge_classes - 1, dtype=torch.long
        )
        for (src, dst), bond_type in zip(bond_idxs, bond_types):
            edge_class_matrix[src, dst] = bond_type
            edge_class_matrix[dst, src] = (
                bond_type  # Ensure symmetry by assigning both directions
            )

        # Check for symmetry
        assert torch.equal(
            edge_class_matrix, edge_class_matrix.T
        ), "edge_class_matrix is not symmetric"

        edge_features = F.one_hot(
            edge_class_matrix, num_classes=self.edge_classes
        )  # Shape: [num_nodes, num_nodes, edge_classes]

        # Create the PyTorch Geometric Data object (graph)
        edge_index = torch.tensor(bond_idxs).T  # Shape: [2, num_edges]

        # Flatten edge feature matrix for each edge: [num_edges, edge_classes]
        edge_attr = edge_features[
            edge_index[0], edge_index[1]
        ]  # Shape: [num_edges, edge_classes]

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        return data


def create_dataloader(data, batch_size=32, nodes_classes=5, edge_classes=4):
    # Create the dataset
    dataset = QM9GraphDataset(data, nodes_classes, edge_classes)

    # Create the DataLoader (PyTorch Geometric handles variable graph sizes)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return dataloader
