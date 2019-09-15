import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class RetDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.ret_pairs = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ret_pairs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.ret_pairs.iloc[idx, 0]) + ".jpeg")
        image = Image.open(img_name)
        label = self.ret_pairs.iloc[idx, 1]
        sample = [image, label]

        if self.transform:
            sample[0] = self.transform(sample[0])

        return sample

    # assumes 5 classes only
    def get_ratios(self):
        label_counts = np.zeros(5)
        for idx in range(self.__len__()):
            label = self.ret_pairs.iloc[idx, 1]
            label_counts[label] += 1
        ratios = np.empty(5)
        for i in range(5):
            ratios[i] = label_counts[i] / sum(label_counts)
        return ratios, label_counts

    def get_img_names(self):
        arr = np.asarray(self.ret_pairs)
        return arr[:, 0]

    def get_labels(self):
        arr = np.asarray(self.ret_pairs)
        return arr[:, 1]
