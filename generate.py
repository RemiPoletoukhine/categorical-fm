import yaml
import torch
import numpy as np
from train import load_qm9
from logger import set_logger
from src.qm9 import qm9_dataset
from src.catflow.catflow import CatFlow
from src.catflow.utils import get_device
from src.qm9.generation_utils import check_validity, check_novelty

# NOTE: adapted from https://github.com/calvin-zcx/moflow


if __name__ == "__main__":
    # Initialize logger
    logger = set_logger("generation")
    # read the config file
    config = yaml.safe_load(open("configs/catflow.yaml", "r"))
    # read the qm9 config file
    qm9_config = yaml.safe_load(open("configs/qm9.yaml", "r"))
    # set the device
    device = get_device()
    logger.info(f"Device used: {device}")
    # Prepare the qm9 dataset
    datamodule, dataset_infos, domain_features = load_qm9(qm9_config)
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule, domain_features=domain_features
    )

    # loading the model. Currently, loads CatFlow always.
    model = CatFlow(config, dataset_infos, domain_features, device).to(device)
    # load the trained weights and set the model to evaluation mode
    model.load_state_dict(
        torch.load("model_dicts/catflow_best_2310.pt", weights_only=True)
    )
    logger.info("Model loaded successfully.")
    model.eval()

    # set the atomic number list and get the smiles from the training dataset
    atomic_num_list = [6, 7, 8, 9]
    train_smiles = qm9_dataset.get_train_smiles(
        cfg=qm9_config,
        train_dataloader=datamodule.train_dataloader(),
        dataset_infos=dataset_infos,
        evaluate_dataset=False,
    )
    valid_ratio, unique_ratio, novel_ratio = [], [], []
    abs_unique_ratio, abs_novel_ratio = [], []
    for i in range(config["n_iter"]):
        # Generate 10k molecules
        logger.info("Starting the generation of the graphs.")
        # Start the generation of the molecules
        nodes_repr, edges_repr = model.sampling(
            num_nodes=config["n_nodes"],
        )  # output: (batch_size, 9, 4), (batch_size, 9, 9, 5)
        val_res = check_validity(
            edges_repr, nodes_repr, atomic_num_list, correct_validity=True
        )
        novel_r, abs_novel_r = check_novelty(
            val_res["valid_smiles"], train_smiles, nodes_repr.shape[0]
        )
        novel_ratio.append(novel_r)
        abs_novel_ratio.append(abs_novel_r)

        unique_ratio.append(val_res["unique_ratio"])
        abs_unique_ratio.append(val_res["abs_unique_ratio"])
        valid_ratio.append(val_res["valid_ratio"])
        n_valid = len(val_res["valid_mols"])

    logger.info(
        "validity: mean={:.2f}%, sd={:.2f}%, vals={}".format(
            np.mean(valid_ratio), np.std(valid_ratio), valid_ratio
        )
    )
    logger.info(
        "novelty: mean={:.2f}%, sd={:.2f}%, vals={}".format(
            np.mean(novel_ratio), np.std(novel_ratio), novel_ratio
        )
    )
    logger.info(
        "uniqueness: mean={:.2f}%, sd={:.2f}%, vals={}".format(
            np.mean(unique_ratio), np.std(unique_ratio), unique_ratio
        )
    )
    logger.info(
        "abs_novelty: mean={:.2f}%, sd={:.2f}%, vals={}".format(
            np.mean(abs_novel_ratio), np.std(abs_novel_ratio), abs_novel_ratio
        )
    )
    logger.info(
        "abs_uniqueness: mean={:.2f}%, sd={:.2f}%, vals={}".format(
            np.mean(abs_unique_ratio), np.std(abs_unique_ratio), abs_unique_ratio
        )
    )
