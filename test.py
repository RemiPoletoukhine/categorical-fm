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
import sys

from src.catflow.statflow import GraphStatFlow, GraphStatFlow_Simplex

def load_qm9(qm9_config):
    datamodule = qm9_dataset.QM9DataModule(qm9_config)
    dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=qm9_config)
    extra_features = ExtraFeatures('all', dataset_info=dataset_infos)
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

    return datamodule, dataset_infos, extra_features, domain_features


logger = set_logger("train")
# read the config file
config = yaml.safe_load(open("configs/catflow.yaml", "r"))
# read the qm9 config file
qm9_config = yaml.safe_load(open("configs/qm9.yaml", "r"))
# set the device
device = torch.device('cpu') #get_device()
logger.info(f"Device used: {device}")
# Prepare the qm9 dataset
datamodule, dataset_infos, extra_features, domain_features = load_qm9(qm9_config)
dataset_infos.compute_input_output_dims(
    datamodule=datamodule, extra_features=extra_features, domain_features=domain_features
)
model = GraphStatFlow_Simplex(config, dataset_infos, domain_features, device).to(device)
sd = torch.load('fuckedmodel1.pt')
model.load_state_dict(sd)
samples = model.sample('ode', 9, 1, device, None)
print(samples[0][0].size(), samples[0][1].size())