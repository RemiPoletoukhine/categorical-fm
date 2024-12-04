from src.catflow.utils import to_dense, get_device, get_writer, EarlyStopper, PlaceHolder
from src.catflow.dataset import create_dataloader
from torch_ema import ExponentialMovingAverage
from torch_geometric.loader import DataLoader
from src.dirichletFlow.flow_utils import sample_cond_prob_path
from src.dirichletFlow.dirichlet import DirichletFlow
from src.metrics.metrics import TrainLossDiscrete
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import argparse
import logging
import torch
import yaml
import os
import torch_geometric

from src.qm9.extra_features_molecular import ExtraMolecularFeatures
from src.qm9.extra_features import ExtraFeatures
from src.qm9 import qm9_dataset
from logger import set_logger
from src.dirichletFlow.flow_utils import expand_simplex


def load_qm9(qm9_config):
    datamodule = qm9_dataset.QM9DataModule(qm9_config)
    dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=qm9_config)
    extra_features = ExtraFeatures('all', dataset_info=dataset_infos)
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

    return datamodule, dataset_infos, extra_features, domain_features


def step_forward(
    model: DirichletFlow,
    optimizer: torch.optim.Optimizer,
    data,
    criterion: TrainLossDiscrete,
    device: torch.device,
    config: dict
):
    """
    Perform a forward pass of the model and compute the loss.

    Args:
        model (CatFlow): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        data: The batch to train on.
        criterion (TrainLossDiscrete): The loss function to use.
        device (torch.device): The device to use.

    """
    if data.edge_index.numel() == 0:
        logger.info("Found a batch with no edges. Skipping.")
        return
    data = data.to(device)
    batch_size = len(data)
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
    # Zero the gradients if training
    if optimizer:
        optimizer.zero_grad()
    # CatFlow forward pass:
    # Step 1: Sample t , x , e 
    x_t,t = sample_cond_prob_path(x_1, config["n_node_classes"],None)  
    #x_t, prior_weights_node = expand_simplex(x_t,t)
    B,l,l,n = e_1.shape
    e_1 = e_1.reshape(B,l*l,n)
    e_t, _ = sample_cond_prob_path(e_1, config["n_edge_classes"],t)
    #e_t, prior_weights_edge = expand_simplex(e_t,t)
    e_t = e_t.reshape(B,l,l,n)

    # Need to take care of symmetrical property of edges.
    upper_triangular_mask = torch.zeros_like(e_t)
    indices = torch.triu_indices(row=e_t.size(1), col=e_t.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    e_t = e_t * upper_triangular_mask
    e_t = e_t + torch.transpose(e_t, 1, 2)

    assert (e_t == torch.transpose(e_t, 1, 2)).all()
    # Perform masking
    z_t = PlaceHolder(X=x_t, E=e_t, y=y_1).type_as(x_t).mask(node_mask)
    noisy_data = {
        "X_t": z_t.X,
        "E_t": z_t.E,
        "y_t": z_t.y,
        "node_mask": node_mask.type(torch.bool),
        "t": t,
    }
    # compute the extra molecular features
    extra_data = model.compute_extra_data(noisy_data=noisy_data)
    # Step 3: Forward pass of the graph transformer
    theta_pred = model.forward(
        t=t, noisy_data=noisy_data, extra_data=extra_data, node_mask=node_mask
    )
    # Step 4: Calculate the loss
    loss = criterion(
        masked_pred_X=theta_pred.X,
        masked_pred_E=theta_pred.E,
        pred_y=theta_pred.y,
        true_X=x_1,
        true_E=e_1,
        true_y=y_1,
        log=False,
    )

    return loss



def train_epoch(
    model: DirichletFlow,
    optimizer: torch.optim.Optimizer,
    dataloader: torch_geometric.loader.dataloader.DataLoader,
    criterion: TrainLossDiscrete,
    ema: ExponentialMovingAverage,
    device: torch.device,
    config: dict
):
    """
    Train the model for one epoch.

    Args:
        model (CatFlow): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        dataloader (torch_geometric.loader.DataLoader): The dataloader to use.
        criterion (TrainLossDiscrete): The loss function to use.
        ema (ExponentialMovingAverage): The EMA to use.
        device (torch.device): The device to use.
    """
    model.train()
    total_loss = 0
    for data in tqdm(dataloader):
        # Steps 1-4: Forward pass
        loss = step_forward(model, optimizer, data, criterion, device,config)
        # Step 5: Backward pass
        loss.backward()
        optimizer.step()
        # Step 6: Update the EMA
        ema.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device,config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            loss = step_forward(model, None, data, criterion, device,config)
            total_loss += loss.item()

    return total_loss / len(dataloader)



def training(
    model: DirichletFlow,
    optimizer: optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    criterion: nn.Module,
    ema: ExponentialMovingAverage,
    datamodule: qm9_dataset.QM9DataModule,
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

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    len_train_dataloader = len(train_dataloader)
    len_val_dataloader = len(val_dataloader)

    writer.add_custom_scalars(layout)
    min_val_loss = float("inf")
    # initialise early stopper
    early_stopper = EarlyStopper(patience=3, min_delta=0.005)
    for epoch in range(config["n_epochs"]):
        train_loss = train_epoch(
            model, optimizer, train_dataloader, criterion, ema, device,config
        )
        writer.add_scalar(
            f"loss/train",
            train_loss,
            (epoch + 1) * len_train_dataloader,
        )
        val_loss = validate_epoch(model, val_dataloader, criterion, device,config)
        writer.add_scalar(
            f"loss/validation",
            val_loss,
            (epoch + 1) * len_train_dataloader + len_val_dataloader,
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
            torch.save(model.state_dict(), f"model_dicts/dirichletFlow_best.pt")
        # early stopping if the validation loss does not decrease
        if early_stopper.early_stop(val_loss):
            logging.info("Early stopping triggered.")
            break

        scheduler.step()

    logging.info(f"Training complete. Best epoch: {best_epoch}")

    return 0



if __name__ == "__main__":
    logger = set_logger("train")
    # read the config file
    config = yaml.safe_load(open("configs/catflow.yaml", "r"))
    # read the qm9 config file
    qm9_config = yaml.safe_load(open("configs/qm9.yaml", "r"))
    # set the device
    device = get_device()
    logger.info(f"Device used: {device}")
    # Prepare the qm9 dataset
    datamodule, dataset_infos, extra_features, domain_features = load_qm9(qm9_config)
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule, extra_features=extra_features,domain_features=domain_features
    )
    # Define the model
    model = DirichletFlow(config,dataset_infos, domain_features, device).to(device)

    # Define the optimizer, scheduler and loss function
    logger.info("Starting the training.")
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
    criterion = TrainLossDiscrete(lambda_train=config["lambda_train"]).to(device)
    # Define the EMA
    ema = ExponentialMovingAverage(model.parameters(), decay=config["ema_decay"])

    # set the seed
    torch.manual_seed(config["seed"])

    # train the model
    _ = training(
        model,
        optimizer,
        scheduler,
        criterion,
        ema,
        datamodule,
        device,
        config,
    )
