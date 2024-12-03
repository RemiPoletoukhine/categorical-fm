import os
import yaml
import torch
import datetime
import numpy as np
from train import load_qm9
from logger import set_logger
from src.qm9 import qm9_dataset
from src.catflow.catflow import CatFlow
from src.dirichletFlow.dirichlet import DirichletFlow
from src.dirichletFlow.flow_utils import expand_simplex
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
    datamodule, dataset_infos, extra_features, domain_features = load_qm9(qm9_config)
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule, extra_features=extra_features, domain_features=domain_features
    )

    # loading the model. Currently, loads CatFlow always.
    model = DirichletFlow(config, dataset_infos, domain_features, device).to(device)
    # load the trained weights and set the model to evaluation mode
    model.load_state_dict(
        torch.load("model_dicts/dirichletFlow_best.pt", weights_only=True)
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
            f"Generated {n_valid} valid molecules out of {nodes_repr.shape[0]}."
        )
        
        # Save the generated molecules
        os.makedirs("generated", exist_ok=True)
        time = datetime.datetime.now()
        time = str(time).replace(":","_")
        np.save(f"generated/graphs_{i}_{time}.npy", nodes_repr.detach().cpu().numpy())
        np.save(f"generated/edges_{i}_{time}.npy", edges_repr.detach().cpu().numpy())
        logger.info(f"Generated molecules saved for iteration {i}.")
        # Log the current statistics
        logger.info(
            f"validity: mean={np.mean(valid_ratio):.2f}%"
        )
        logger.info(
            f"novelty: mean={np.mean(novel_ratio):.2f}%"
        )
        logger.info(
            f"uniqueness: mean={np.mean(unique_ratio):.2f}%"
        )
        logger.info(
            f"abs_novelty: mean={np.mean(abs_novel_ratio):.2f}%"
        )
        logger.info(
            f"abs_uniqueness: mean={np.mean(abs_unique_ratio):.2f}%"
        )

    logger.info(
        f"validity: mean={np.mean(valid_ratio):.2f}%, sd={np.std(valid_ratio):.2f}%, vals={valid_ratio}"
    )
    logger.info(
        f"novelty: mean={np.mean(novel_ratio):.2f}%, sd={np.std(novel_ratio):.2f}%, vals={novel_ratio}"
    )
    logger.info(
        f"uniqueness: mean={np.mean(unique_ratio):.2f}%, sd={np.std(unique_ratio):.2f}%, vals={unique_ratio}"
    )
    logger.info(
        f"abs_novelty: mean={np.mean(abs_novel_ratio):.2f}%, sd={np.std(abs_novel_ratio):.2f}%, vals={abs_novel_ratio}"
    )
    logger.info(
        f"abs_uniqueness: mean={np.mean(abs_unique_ratio):.2f}%, sd={np.std(abs_unique_ratio):.2f}%, vals={abs_unique_ratio}"
    )
