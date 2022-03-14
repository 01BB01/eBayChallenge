import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class eBayDataset(Dataset):
    def __init__(self, split_name, transform=None, root_dir="./data/eBay/"):
        super().__init__()
        csv_file = split_name + ".csv"
        self.root_dir = root_dir
        self.split_name = split_name
        self.annotations = pd.read_csv(os.path.join(root_dir, "metadata", csv_file)).to_dict()
        self.transform = transform

    def __len__(self):
        if self.split_name == "index":
            return 5000
        return len(self.annotations["UUID"])

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "Images", self.annotations["IMAGE_PATH"][idx])
        image = Image.open(img_name)
        if self.transform is not None:
            image = self.transform(image)

        sample = {"image": image, "uuid": self.annotations["UUID"][idx], "id": idx}
        if self.split_name in ["train", "val"]:
            sample["label_1"] = self.annotations["META_CATEG_ID"][idx]
            sample["label_2"] = self.annotations["CATEG_LVL2_ID"][idx]
            sample["label_3"] = self.annotations["LEAF_CATEG_ID"][idx]
            sample["text"] = self.annotations["AUCT_TITL"][idx]

        return sample
