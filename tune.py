from train import train_epoch, validate_epoch, prepare_dataloaders
from torch_ema import ExponentialMovingAverage
from torch_geometric.loader import DataLoader
from src.catflow.utils import get_device
from src.catflow.catflow import CatFlow
from optuna.trial import TrialState
from datetime import datetime
from functools import partial
import torch.optim as optim
import torch.nn as nn
import logging
import optuna
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
            f"code_logs/tuning_{datetime.now()}.log", mode="w"
        ),  # Log to a file
    ],
)


def objective(
    trial,
    model: CatFlow,
    criterion: nn.Module,
    ema: ExponentialMovingAverage,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    config: dict,
):
    """
    Objective function for Optuna to optimize the hyperparameters of the model.
    """
    n_epochs = config["n_epochs_tuning"]
    # tune the hyperparameters: optimizer and learning rate
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "RMSprop"])
    lr = 10 ** trial.suggest_float("log_lr", -5, -2)
    optimizer = getattr(optim, optimizer_name)(
        model.parameters(), lr=lr, weight_decay=float(config['weight_decay'])
    )
    if config["scheduler"] == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )
    else:
        raise ValueError(f"Invalid scheduler: {config['scheduler']}")
    try:
        for epoch in range(n_epochs):
            train_loss = train_epoch(
                model, optimizer, train_dataloader, criterion, ema, device
            )
            val_loss = validate_epoch(model, val_dataloader, criterion, device)
            scheduler.step()
            logging.info(
                f"Epoch {epoch + 1}/{n_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            # we report average validation loss to Optuna
            trial.report(val_loss, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    except Exception as e:
        logging.error(f"Trial failed with exception: {e}")
        # Mark the trial as pruned so Optuna skips it
        raise optuna.exceptions.TrialPruned()
    
    return val_loss

def hyperparam_tuning(
    objective,
    model,
    criterion,
    ema,
    train_dataloader,
    val_dataloader,
    device,
    config,
):
    """
    Hyperparameter tuning for the Catflow model.
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

    logging.info("Study statistics:")
    logging.info("  Number of finished trials: %d", len(study.trials))
    logging.info("  Number of pruned trials: %d", len(pruned_trials))
    logging.info("  Number of complete trials: %d", len(complete_trials))

    logging.info("Best trial:")
    trial = study.best_trial
    logging.info("  Value: %f", trial.value)

    logging.info("  Params:")
    for key, value in trial.params.items():
        logging.info("    %s: %s", key, value)

    return 0


if __name__ == "__main__":
    # read the config file
    config = yaml.safe_load(open("configs/catflow.yaml", "r"))
    # set the device
    device = get_device()
    # Define the model
    model = CatFlow(config, device).to(device)
    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()
    # Define the EMA
    ema = ExponentialMovingAverage(model.parameters(), decay=config["ema_decay"])
    # set the seed
    torch.manual_seed(config["seed"])
    # prepare the dataloaders
    train_dataloader, val_dataloader = prepare_dataloaders(config["batch_size"])
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
