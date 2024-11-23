from train import train_epoch, validate_epoch, load_qm9
from src.metrics.metrics import TrainLossDiscrete
from torch_ema import ExponentialMovingAverage
from torch_geometric.loader import DataLoader
from src.catflow.utils import get_device
from src.catflow.catflow import CatFlow
from optuna.trial import TrialState
from logger import set_logger
from datetime import datetime
from functools import partial
import torch.optim as optim
import torch.nn as nn
import optuna
import torch
import yaml
import os


def objective(
    trial: optuna.Trial,
    model: CatFlow,
    criterion: TrainLossDiscrete,
    ema: ExponentialMovingAverage,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    config: dict,
):
    """
    Objective function for Optuna to optimize the hyperparameters of the model.

    Args:
        trial (optuna.Trial): The current trial.
        model (CatFlow): The CatFlow model.
        criterion (TrainLossDiscrete): The loss function.
        ema (ExponentialMovingAverage): The Exponential Moving Average.
        train_dataloader (DataLoader): The training DataLoader.
        val_dataloader (DataLoader): The validation DataLoader.
        device (torch.device): The device to use.
        config (dict): The configuration dictionary.
    """
    n_epochs = config["n_epochs_tuning"]
    # tune the hyperparameters: optimizer and learning rate
    optimizer_name = trial.suggest_categorical(
        "optimizer", ["AdamW", "Adam"]
    )
    lr = 10 ** trial.suggest_float("log_lr", -5, -2)
    optimizer = getattr(optim, optimizer_name)(
        model.parameters(), lr=lr, weight_decay=float(config["weight_decay"])
    )
    if config["scheduler"] == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )
    else:
        raise ValueError(f"Invalid scheduler: {config['scheduler']}")
    for epoch in range(n_epochs):
        train_loss = train_epoch(
            model, optimizer, train_dataloader, criterion, ema, device
        )
        val_loss = validate_epoch(model, val_dataloader, criterion, device)
        scheduler.step()
        logger.info(
            f"Epoch {epoch + 1}/{n_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        # we report average validation loss to Optuna
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


def hyperparam_tuning(
    objective: callable,
    model: CatFlow,
    criterion: TrainLossDiscrete,
    ema: ExponentialMovingAverage,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    config: dict,
):
    """
    Hyperparameter tuning for the Catflow model.

    Args:
        objective (function): The objective function.
        model (CatFlow): The CatFlow model.
        criterion (TrainLossDiscrete): The loss function.
        ema (ExponentialMovingAverage): The Exponential Moving Average.
        train_dataloader (DataLoader): The training DataLoader.
        val_dataloader (DataLoader): The validation DataLoader.
        device (torch.device): The device to use.
        config (dict): The configuration dictionary.
    """

    objective = partial(
        objective,
        model=model,
        criterion=criterion,
        ema=ema,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        config=config,
    )

    os.makedirs("optuna_logs", exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///optuna_logs/Trial_{datetime.now()}.sqlite3",
        study_name="CatFlow",
    )
    study.optimize(objective, n_trials=100, timeout=150000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("Study statistics:")
    logger.info("  Number of finished trials: %d", len(study.trials))
    logger.info("  Number of pruned trials: %d", len(pruned_trials))
    logger.info("  Number of complete trials: %d", len(complete_trials))

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info("  Value: %f", trial.value)

    logger.info("  Params:")
    for key, value in trial.params.items():
        logger.info("    %s: %s", key, value)

    return 0


if __name__ == "__main__":
    # Initialize logger
    logger = set_logger("tune")
    # read the config file
    config = yaml.safe_load(open("configs/catflow.yaml", "r"))
    # read the qm9 config file
    qm9_config = yaml.safe_load(open("configs/qm9.yaml", "r"))
    # set the device
    device = get_device()
    logger.info(f"Device used: {device}")
    # Prepare the qm9 dataset and dataloaders
    datamodule, dataset_infos, domain_features = load_qm9(qm9_config)
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule, domain_features=domain_features
    )
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    # Define the model
    model = CatFlow(config, dataset_infos, domain_features, device).to(device)
    # Define the loss function
    criterion = TrainLossDiscrete(lambda_train=config["lambda_train"]).to(device)
    # Define the EMA
    ema = ExponentialMovingAverage(model.parameters(), decay=config["ema_decay"])
    # set the seed
    torch.manual_seed(config["seed"])
    # tune the hyperparameters
    _ = hyperparam_tuning(
        objective,
        model,
        criterion,
        ema,
        train_dataloader,
        val_dataloader,
        device,
        config,
    )
