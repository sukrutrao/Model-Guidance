import numpy as np
import os
import torch
import utils


class VOCDetectParsed(torch.utils.data.Dataset):

    def __init__(self, root, image_set, transform=None, annotated_fraction=1.0):
        super().__init__()
        data_dict = torch.load(os.path.join(root, image_set + ".pt"))
        self.data = data_dict["data"]
        self.labels = data_dict["labels"]
        self.bbs = data_dict["bbs"]
        assert len(self.data) == len(self.labels)
        assert len(self.data) == len(self.bbs)
        self.annotated_fraction = annotated_fraction
    
        if self.annotated_fraction < 1.0:
            annotated_indices = np.random.choice(len(self.bbs), size=int(
                self.annotated_fraction*len(self.bbs)), replace=False)
            for bb_idx in range(len(self.bbs)):
                if bb_idx not in annotated_indices:
                    self.bbs[bb_idx] = None
        self.transform = transform

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.data[idx]), self.labels[idx], self.bbs[idx]
        return self.data[idx], self.labels[idx], self.bbs[idx]

    def load_data(self, idx, pred_class):
        img, labels, bbs = self.__getitem__(idx)
        label = labels[pred_class]
        bb = utils.filter_bbs(bbs, pred_class)
        return img, label, bb

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        data = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        bbs = [item[2] for item in batch]
        return data, labels, bbs