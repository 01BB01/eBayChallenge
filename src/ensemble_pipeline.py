import os

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src import utils
from src.utils.post_processing import re_ranking, whitening

log = utils.get_logger(__name__)


def check_dir(dir):
    if not os.path.isabs(dir):
        return os.path.join(hydra.utils.get_original_cwd(), dir)
    else:
        return dir


def reorder(uuid, feats):
    return sorted(uuid), feats[sorted(range(len(uuid)), key=lambda k: uuid[k])]


def ensemble(config: DictConfig) -> None:

    log.info("Starting ensemble!")

    log.info("Loading features...")
    query_feats_list = []
    index_feats_list = []
    for dir in config.feature_dirs:
        query_data = torch.load(os.path.join(check_dir(dir), "query_feats_and_uuid.pth"))
        index_data = torch.load(os.path.join(check_dir(dir), "index_feats_and_uuid.pth"))

        query_uuid = query_data["query_uuid"]
        query_feats = query_data["query_feats"]
        query_uuid, query_feats = reorder(query_uuid, query_feats)

        index_uuid = index_data["index_uuid"]
        index_feats = index_data["index_feats"]
        index_uuid, index_feats = reorder(index_uuid, index_feats)

        if config.get("whitening"):
            query_feats, index_feats = whitening(query_feats, index_feats)

        query_feats = F.normalize(query_feats, p=2, dim=-1)
        index_feats = F.normalize(index_feats, p=2, dim=-1)
        query_feats_list.append(query_feats)
        index_feats_list.append(index_feats)

    query_feats = torch.cat(query_feats_list, dim=1)
    index_feats = torch.cat(index_feats_list, dim=1)

    dist_matrix = []
    chunk_size = 5000

    for i in range(index_feats.shape[0] // chunk_size + 1):
        print(f"{i}/{index_feats.shape[0] // chunk_size}", end="\r")
        start = i * chunk_size
        end = start + chunk_size
        cdist = query_feats @ index_feats[start:end].t()
        dist_matrix.append(cdist)

    dist_matrix = torch.cat(dist_matrix, dim=1)  # q * i

    if config.get("re_ranking"):
        log.info("Applying re-ranking to cosine distance...")
        _, pred_indices = torch.topk(dist_matrix, k=100, dim=1, largest=True, sorted=True)
        topk_index_feats = index_feats[pred_indices]

        g_g_dist = 2 - 2 * (topk_index_feats @ topk_index_feats.transpose(1, 2))
        q_g_dist = 2 - 2 * (query_feats.unsqueeze(1) @ topk_index_feats.transpose(1, 2))
        q_g_dist = q_g_dist.squeeze(1)
        q_q_dist = 2 - 2 * query_feats @ query_feats.t()

        new_dists = []
        for qidx in range(query_feats.size(0)):
            print(qidx, end="\r")
            new_dists.append(
                re_ranking(
                    q_g_dist[qidx].view(1, -1),
                    q_q_dist[qidx, qidx].view(1, 1),
                    g_g_dist[qidx],
                )
            )
        new_dists = np.stack(new_dists)
        rerank_dists = torch.from_numpy(new_dists).squeeze()
        _, rerank_indices = torch.topk(rerank_dists, k=10, dim=1, largest=False, sorted=True)
        pred_indices = torch.gather(pred_indices, 1, rerank_indices)
    else:
        _, pred_indices = torch.topk(dist_matrix, k=10, dim=1, largest=True, sorted=True)

    log.info("Writing predictions csv!")
    df = pd.DataFrame(zip(query_uuid, pred_indices.cpu().numpy()))
    df[1] = df[1].apply(lambda x: " ".join([index_uuid[i] for i in x]))
    df.to_csv(
        os.path.join(check_dir(config.csv_save_dir), "ensemble_predictions.csv"),
        index=False,
        header=False,
    )
