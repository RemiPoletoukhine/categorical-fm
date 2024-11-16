from src.catflow.utils import to_dense, get_device, get_writer, EarlyStopper
from src.catflow.dataset import create_dataloader
from torch_ema import ExponentialMovingAverage
from torch_geometric.loader import DataLoader
from src.catflow.catflow import CatFlow
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import argparse
import logging
import torch
import yaml
import os

# Configure logging
os.makedirs("code_logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format for log messages
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(
            f"code_logs/training_{datetime.now()}.log", mode="w"
        ),  # Log to a file
    ],
)


def prepare_dataloaders(
    batch_size: int,
    path_to_train: str = "qm9/train_data_processed.pt",
    path_to_val: str = "qm9/val_data_processed.pt",
) -> tuple[DataLoader, DataLoader]:

    # Load your processed data
    train_data = torch.load(path_to_train, weights_only=True)
    val_data = torch.load(path_to_val, weights_only=True)

    # Create the DataLoaders
    train_dataloader = create_dataloader(train_data, batch_size=batch_size)
    val_dataloader = create_dataloader(val_data, batch_size=batch_size)

    return train_dataloader, val_dataloader


def train_epoch(model, optimizer, dataloader, criterion, ema, device):
    model.train()
    total_loss = 0
    for data in tqdm(dataloader):
        data = data.to(device)
        # Get the dense representation of the graph
        dense_data, node_mask = to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        x_true, e_true = dense_data.X.to(device), dense_data.E.to(device)
        node_mask = node_mask.to(device)
        num_nodes = x_true.size(1)
        # Zero the gradients
        optimizer.zero_grad()
        # CatFlow forward pass
        # Step 1: Sample t ~ Exp(1), x ~ N(0, I), e ~ N(0, I)
        t = model.sample_time().to(device)
        x = model.sample_noise(kind="node", num_nodes=num_nodes).to(device)
        e = model.sample_noise(kind="edge", num_nodes=num_nodes).to(device)
        # Step 2: Forward pass of the graph transformer
        inferred = model.forward(t=t, x=x, e=e, node_mask=node_mask)
        # Step 3: Calculate the loss
        loss = criterion(inferred.X, x_true.float()) + criterion(
            inferred.E, e_true.float()
        )
        # Step 4: Backward pass
        loss.backward()
        optimizer.step()
        # Step 5: Update the EMA
        ema.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            data = data.to(device)
            # Get the dense representation of the graph
            dense_data, node_mask = to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            x_true, e_true = dense_data.X.to(device), dense_data.E.to(device)
            node_mask = node_mask.to(device)
            num_nodes = x_true.size(1)
            # CatFlow forward pass
            # Step 1: Sample t ~ Exp(1), x ~ N(0, I), e ~ N(0, I)
            t = model.sample_time().to(device)
            x = model.sample_noise(kind="node", num_nodes=num_nodes).to(device)
            e = model.sample_noise(kind="edge", num_nodes=num_nodes).to(device)
            # Step 2: Forward pass of the graph transformer
            inferred = model.forward(t=t, x=x, e=e, node_mask=node_mask)
            # Step 3: Calculate the loss
            loss = criterion(inferred.X, x_true.float()) + criterion(
                inferred.E, e_true.float()
            )
            total_loss += loss.item()

    return total_loss / len(dataloader)


def training(
    model: CatFlow,
    optimizer: optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    criterion: nn.Module,
    ema: ExponentialMovingAverage,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    config: dict,
) -> None:

    # initialise the SummaryWriter
    writer = get_writer()
    layout = {
        "Test": {
            "ce_loss": ["Multiline", ["loss/train", "loss/validation"]],
        },
    }
    writer.add_custom_scalars(layout)
    min_val_loss = float("inf")
    # initialise early stopper
    early_stopper = EarlyStopper(patience=3, min_delta=0.005)
    for epoch in range(config["n_epochs"]):
        train_loss = train_epoch(
            model, optimizer, train_dataloader, criterion, ema, device
        )
        writer.add_scalar(
            f"loss/train",
            train_loss,
            (epoch + 1) * len(train_dataloader),
        )
        val_loss = validate_epoch(model, val_dataloader, criterion, device)
        writer.add_scalar(
            f"loss/validation",
            val_loss,
            (epoch + 1) * len(train_dataloader) + len(val_dataloader),
        )
        logging.info(
            f"Epoch {epoch + 1}/{config['n_epochs']}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < min_val_loss:
            logging.info(
                f"Validation loss decreased from {min_val_loss:.4f} to {val_loss:.4f}. Saving model."
            )
            min_val_loss = val_loss
            best_epoch = epoch
            # Save the model
            os.makedirs("model_dicts", exist_ok=True)
            torch.save(model.state_dict(), f"model_dicts/catflow_best.pt")
        # early stopping if the validation loss does not decrease
        if early_stopper.early_stop(val_loss):
            logging.info("Early stopping triggered.")
            break

        scheduler.step()

    logging.info(f"Training complete. Best epoch: {best_epoch}")

    return 0


def inference(model, config, device):
    # Step 1: x ~ N(0, I), e ~ N(0, I) and initialize the node mask
    x = model.sample_noise(kind="node", num_nodes=config["n_nodes"]).to(device)
    e = model.sample_noise(kind="edge", num_nodes=config["n_nodes"]).to(device)
    node_mask = torch.ones(config["batch_size"], config["n_nodes"]).to(device)
    # Step 2: initialize the state
    init_state = (x, e, node_mask)
    # Step 3: sample from the model
    nodes_repr, edges_repr = model.sampling(init_state)

    return nodes_repr, edges_repr


if __name__ == "__main__":
    # read the config file
    config = yaml.safe_load(open("configs/catflow.yaml", "r"))
    # set the device
    device = get_device()
    # Define the model
    model = CatFlow(config, device).to(device)

    parser = argparse.ArgumentParser("catflow_script")
    parser.add_argument(
        "mode", help="1: inference mode, other int: training mode", type=int
    )
    args = parser.parse_args()
    if args.mode:
        start = datetime.now()
        logging.info("Starting the generation of the graphs.")
        nodes_repr, edges_repr = inference(model, config, device)
        os.makedirs("generated_graphs", exist_ok=True)
        torch.save(nodes_repr, "generated_graphs/nodes_repr.pt")
        torch.save(edges_repr, "generated_graphs/edges_repr.pt")
        logging.info(
            f"Graphs generated successfully in: {datetime.now() - start} seconds."
        )
    else:
        logging.info("Starting the training.")
        # Define the optimizer, scheduler and loss function
        if config["optimizer"] == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config["lr"],
                weight_decay=float(config["weight_decay"]),
            )
        else:
            raise ValueError(f"Invalid optimizer: {config['optimizer']}")
        if config["scheduler"] == "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["n_epochs"]
            )
        else:
            raise ValueError(f"Invalid scheduler: {config['scheduler']}")
        criterion = nn.BCEWithLogitsLoss()
        # Define the EMA
        ema = ExponentialMovingAverage(model.parameters(), decay=config["ema_decay"])

        # set the seed
        torch.manual_seed(config["seed"])
        # prepare the dataloaders
        train_dataloader, val_dataloader = prepare_dataloaders(config["batch_size"])
        # train the model
        _ = training(
            model,
            optimizer,
            scheduler,
            criterion,
            ema,
            train_dataloader,
            val_dataloader,
            device,
            config,
        )
