import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class eBayDataset(Dataset):
    def __init__(
        self,
        split_name,
        transform=None,
        root_dir="./data/eBay/",
        aug_transform=None,
        multi_label=False,
    ):
        super().__init__()
        csv_file = split_name + ".csv"
        self.root_dir = root_dir
        self.split_name = split_name
        self.annotations = pd.read_csv(os.path.join(root_dir, "metadata", csv_file)).to_dict()
        self.transform = transform
        self.aug_transform = aug_transform
        if multi_label:
            with open(os.path.join(root_dir, split_name + "_multi_labels.json"), "r") as f:
                self.multi_labels = json.load(f)
        else:
            self.multi_labels = None

    def __len__(self):
        return len(self.annotations["UUID"])

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "Images", self.annotations["IMAGE_PATH"][idx])
        image = Image.open(img_name)
        if self.transform is not None:
            original_image = self.transform(image)

        sample = {"image": original_image, "uuid": self.annotations["UUID"][idx], "id": idx}

        if self.aug_transform is not None:
            aug_image = self.aug_transform(image)
            sample["aug_image"] = aug_image

        if self.split_name in ["train", "val"]:
            if self.multi_labels is not None:
                labels = torch.zeros(22295)
                labels[self.multi_labels[idx]] = 1
                sample["multi_label"] = labels
            else:
                sample["label_1"] = self.annotations["META_CATEG_ID"][idx]
                sample["label_2"] = self.annotations["CATEG_LVL2_ID"][idx]
                sample["label_3"] = self.annotations["LEAF_CATEG_ID"][idx]
                sample["text"] = self.annotations["AUCT_TITL"][idx]

        return sample


class eBayRetrievalDataset(eBayDataset):
    def __init__(self, split_name, transform=None, root_dir="./data/eBay/"):
        super().__init__(split_name, transform, root_dir)
        if split_name == "train":
            pseudo_ids = np.load(os.path.join(root_dir, "pseudo_train_ids.npy"))
            self.load_text = True
        elif split_name == "index":
            pseudo_ids = np.load(os.path.join(root_dir, "pseudo_index_ids.npy"))
            self.load_text = False

        self.id_dict = defaultdict(list)
        for i, id in enumerate(pseudo_ids):
            self.id_dict[id].append(i)
        self.pseudo_ids = pseudo_ids

    def _sample_positive(self, idx):
        pseudo_id = self.pseudo_ids[idx]
        candidate = self.id_dict[pseudo_id]
        if len(candidate) == 1:
            return idx
        chosen = np.random.choice(candidate)
        while chosen == idx:
            chosen = np.random.choice(candidate)
        return chosen

    def _get_image(self, idx):
        img_name = os.path.join(self.root_dir, "Images", self.annotations["IMAGE_PATH"][idx])
        image = Image.open(img_name)
        image = self.transform(image)
        return image

    def __getitem__(self, idx):
        image = self._get_image(idx)
        pos_image = self._get_image(self._sample_positive(idx))

        sample = {"image": image, "pos_image": pos_image, "id": self.pseudo_ids[idx]}
        if self.load_text:
            sample["text"] = self.annotations["AUCT_TITL"][idx]

        return sample
