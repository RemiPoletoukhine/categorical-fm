# #%%
# import yaml

# import torch as pt
# from torch.optim import Adam


# from src.catflow.utils import (
#     to_dense,
#     get_device,
#     get_writer,
#     EarlyStopper,
#     PlaceHolder,
# )
# from src.qm9 import qm9_dataset
# from src.qm9.extra_features_molecular import ExtraMolecularFeatures

# def load_qm9(qm9_config):
#     datamodule = qm9_dataset.QM9DataModule(qm9_config)
#     dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=qm9_config)
#     domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

#     return datamodule, dataset_infos, domain_features

# with open("configs/qm9.yaml", "r") as f:
#     qm9_conf = yaml.safe_load(f)

# datamodule,infos,feats = load_qm9(qm9_conf)
# dataloader = datamodule.train_dataloader()
# data_sample = list(dataloader)[0]

# dense_data, node_mask = to_dense(
#         data_sample.x, data_sample.edge_index, data_sample.edge_attr, data_sample.batch
#     )

# #%%



from src.catflow.utils import (
    to_dense,
    get_device,
    get_writer,
    get_writer_windows,
    EarlyStopper,
    PlaceHolder,
)
from src.qm9.extra_features_molecular import ExtraMolecularFeatures
from src.metrics.metrics import TrainLossDiscrete
from src.qm9.extra_features import ExtraFeatures
from torch_ema import ExponentialMovingAverage
from src.catflow.catflow import CatFlow
from src.qm9 import qm9_dataset
from logger import set_logger
from datetime import datetime
import torch.optim as optim
import torch_geometric
from tqdm import tqdm
import torch.nn as nn
import argparse
import einops
import torch
import yaml
import os


from src.catflow.statflow import GraphStatFlow, GraphStatFlow_Simplex

def load_qm9(qm9_config):
    datamodule = qm9_dataset.QM9DataModule(qm9_config)
    dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=qm9_config)
    extra_features = ExtraFeatures('all', dataset_info=dataset_infos)
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

    return datamodule, dataset_infos, extra_features, domain_features


def step_forward(
    model: CatFlow,
    optimizer: torch.optim.Optimizer,
    data,
    criterion: TrainLossDiscrete,
    device: torch.device,
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
    # Step 1: Sample t ~ U[0, 1], x ~ N(0, I), e ~ N(0, I)
    t = model.sample_time(batch_size)
    sampled = model.sample_noise(x=x_1, e=e_1, y=data.y, node_mask=node_mask)
    x_0, e_0, y_0 = sampled.X.to(device), sampled.E.to(device), sampled.y.to(device)
    # Step 2: Compute noisy data x_t from the sampled noise x_0 using linearity assumption
    tau_y, tau_x, tau_e = (
        einops.rearrange(t, "b -> b 1"),
        einops.rearrange(t, "b -> b 1 1"),
        einops.rearrange(t, "b -> b 1 1 1"),
    )
    x_t = tau_x * x_1 + (1 - tau_x) * x_0
    e_t = tau_e * e_1 + (1 - tau_e) * e_0
    y_t = tau_y * y_1 + (1 - tau_y) * y_0
    # Perform masking
    z_t = PlaceHolder(X=x_t, E=e_t, y=y_t).type_as(x_t).mask(node_mask)
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
    model: GraphStatFlow,
    optimizer: torch.optim.Optimizer,
    dataloader: torch_geometric.loader.dataloader.DataLoader,
    ema: ExponentialMovingAverage,
    device: torch.device,
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
        try:
            loss = model.get_loss(data, logger, device, None)
        except AssertionError:
            torch.save(data, 'fuckedbatch.pt')
        # Step 5: Backward pass
        loss.backward()
        optimizer.step()
        # Step 6: Update the EMA
        ema.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(
    model: GraphStatFlow,
    dataloader: torch_geometric.loader.dataloader.DataLoader,
    device: torch.device,
):
    """
    Validate the model for one epoch.

    Args:
        model (CatFlow): The model to validate.
        dataloader (torch_geometric.loader.DataLoader): The dataloader to use.
        criterion (TrainLossDiscrete): The loss function to use.
        device (torch.device): The device to use.
    """

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            loss = model.get_loss(data, logger, device, None)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def training(
    model: GraphStatFlow,
    optimizer: optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    ema: ExponentialMovingAverage,
    datamodule: qm9_dataset.QM9DataModule,
    device: torch.device,
    config: dict,
) -> None:
    """
    Train the model.

    Args:
        model (CatFlow): The model to train.
        optimizer (optim.Optimizer): The optimizer to use.
        scheduler (torch.optim.lr_scheduler.CosineAnnealingLR): The scheduler to use.
        criterion (TrainLossDiscrete): The loss function to use.
        ema (ExponentialMovingAverage): The EMA to use.
        datamodule (qm9_dataset.QM9DataModule): The datamodule to use.
        device (torch.device): The device to use.
        config (dict): The configuration to use.
    """

    # initialise the SummaryWriter
    if os.name == 'nt':
        writer = get_writer_windows()
    else:
        writer = get_writer()

    layout = {
        "Test": {
            "ce_loss": ["Multiline", ["loss/train", "loss/validation"]],
        },
    }
    # define dataloaders and get their number of batches
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    len_train_dataloader = len(train_dataloader)
    len_val_dataloader = len(val_dataloader)

    writer.add_custom_scalars(layout)
    min_val_loss = float("inf")
    # initialise early stopper
    early_stopper = EarlyStopper(patience=50, min_delta=0.005)
    for epoch in range(config["n_epochs"]):
        train_loss = train_epoch(
            model, optimizer, train_dataloader, ema, device
        )
        writer.add_scalar(
            f"loss/train",
            train_loss,
            (epoch + 1) * len_train_dataloader,
        )
        val_loss = validate_epoch(model, val_dataloader, device)
        writer.add_scalar(
            f"loss/validation",
            val_loss,
            (epoch + 1) * len_train_dataloader + len_val_dataloader,
        )
        logger.info(
            f"Epoch {epoch + 1}/{config['n_epochs']}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < min_val_loss:
            logger.info(
                f"Validation loss decreased from {min_val_loss:.4f} to {val_loss:.4f}. Saving model."
            )
            min_val_loss = val_loss
            best_epoch = epoch
            # Save the model
            os.makedirs("model_dicts", exist_ok=True)
            torch.save(model.state_dict(), f"model_dicts/catflow_best.pt")
        # early stopping if the validation loss does not decrease
        if early_stopper.early_stop(val_loss):
            logger.info("Early stopping triggered.")
            break

        scheduler.step()

    logger.info(f"Training complete. Best epoch: {best_epoch}")

    return 0


if __name__ == "__main__":
    # Initialize logger
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
        datamodule=datamodule, extra_features=extra_features, domain_features=domain_features
    )
    # Define the model
    model = GraphStatFlow_Simplex(config, dataset_infos, domain_features, device).to(device)

    # Load epoch 13 model for debugging
    # state_dict = torch.load('model_dicts/catflow_best.pt')
    # model.load_state_dict(state_dict)
    
    parser = argparse.ArgumentParser("catflow_script")
    parser.add_argument(
        "mode", help="1: inference mode, other int: training mode", type=int
    )
    args = parser.parse_args()
    if args.mode:
        start = datetime.now()
        logger.info("Starting the generation of the graphs.")
        # TODO: add loading of the model
        nodes_repr, edges_repr = model.sample(method="ode", num_nodes=dataset_infos.max_n_nodes, n_sample=1, n_steps=0, device=device)
        os.makedirs("generated_graphs", exist_ok=True)
        torch.save(nodes_repr, "generated_graphs/nodes_repr.pt")
        torch.save(edges_repr, "generated_graphs/edges_repr.pt")
        logger.info(
            f"Graphs generated successfully in: {datetime.now() - start} seconds."
        )
    else:
        logger.info("Starting the training.")
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
        # Define the EMA
        ema = ExponentialMovingAverage(model.parameters(), decay=config["ema_decay"])
        # set the seed
        torch.manual_seed(config["seed"])
        # train the model
        _ = training(
            model,
            optimizer,
            scheduler,
            ema,
            datamodule,
            device,
            config,
        )
        
        # totloss = []
        # totloss.append(train_epoch(model, optimizer, datamodule.train_dataloader(), ema, device))
        # totloss.append(train_epoch(model, optimizer, datamodule.train_dataloader(), ema, device))
        # print(*totloss)
