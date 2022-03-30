import os
from typing import List

import hydra
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_logger(__name__)


def gather_filed(predictions):
    feats = []
    uuid = []
    for pred in predictions:
        feats.extend(pred["feats"])
        uuid.extend(pred["uuid"])
    return torch.stack(feats), uuid


def test(config: DictConfig) -> None:
    """Contains minimal example of the testing pipeline.
    Evaluates given checkpoint on a testset.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    if not os.path.isabs(config.ckpt_path):
        config.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), config.ckpt_path)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=logger)

    # Log hyperparameters
    trainer.logger.log_hyperparams({"ckpt_path": config.ckpt_path})

    log.info("Starting testing!")
    query_predictions, index_predictions = trainer.predict(
        model=model, datamodule=datamodule, ckpt_path=config.ckpt_path
    )

    # FIXME: more elegant solution
    query_feats, query_uuid = gather_filed(query_predictions)
    index_feats, index_uuid = gather_filed(index_predictions)

    cdist_matrix = []
    edist_matrix = []
    chunk_size = 5000
    query_feats_norm = F.normalize(query_feats, p=2, dim=-1)
    for i in range(index_feats.shape[0] // chunk_size + 1):
        print(f'{i}/{index_feats.shape[0] // chunk_size}', end='\r')
        start = i * chunk_size
        end = start + chunk_size
        edist = -torch.cdist(query_feats, index_feats[start:end])  # q * 5000
        edist_matrix.append(edist)

        index_feats_norm = F.normalize(index_feats[start:end], p=2, dim=-1)
        cdist = query_feats_norm @ index_feats_norm.t()
        cdist_matrix.append(cdist)

    if not os.path.isabs(config.csv_save_dir):
        config.csv_save_dir = os.path.join(hydra.utils.get_original_cwd(), config.csv_save_dir)

    for prefix, dist_matrix in zip(['cosine', 'euclidean'], [cdist_matrix, edist_matrix]):
        dist_matrix = torch.cat(dist_matrix, dim=1)  # q * i
        _, pred_indices = torch.topk(dist_matrix, k=10, dim=1, largest=True, sorted=True)

        log.info("Writing predictions csv!")
        df = pd.DataFrame(zip(query_uuid, pred_indices.cpu().numpy()))
        df[1] = df[1].apply(lambda x: " ".join([index_uuid[i] for i in x]))
        if not os.path.isabs(config.csv_save_dir):
            config.csv_save_dir = os.path.join(hydra.utils.get_original_cwd(), config.csv_save_dir)
        df.to_csv(os.path.join(config.csv_save_dir, f"{prefix}_predictions.csv"), index=False, header=False)

        if config.get('save_sim'):
            res = {prefix: dist_matrix, 'query_uuid': query_uuid, 'index_uuid': index_uuid}
            torch.save(res, os.path.join(config.csv_save_dir, f"{prefix}_dist_and_uuid.pth"))
