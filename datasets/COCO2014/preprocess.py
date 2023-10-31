import torch
from pycocotools.coco import COCO
from PIL import Image
import os
import torchvision
import argparse
import os
from tqdm import tqdm
from torch.utils.data import Dataset

""" 
Adapted from: https://github.com/stevenstalder/NN-Explainer
"""
class COCODataset(Dataset):
    def __init__(self, root, annotation, transform_fn=None):
        self.root = root
        self.transform = transform_fn
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        num_objects = len(coco_annotation)
        cat_ids = []
        for i in range(num_objects):
            cat_ids.append(coco_annotation[i]['category_id'])

        my_annotation = {}
        my_annotation["width"] = img.size[0]
        my_annotation["height"] = img.size[1]
        my_annotation["bboxes"] = []
        my_annotation
        for annotation in coco_annotation:
            my_annotation["bboxes"].append(
                [annotation["category_id"]]+annotation["bbox"])

        if self.transform is not None:
            img = self.transform(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


def main(args):
    split = args.split
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(
        size=(224, 224)), torchvision.transforms.ToTensor()])
    if split == "train":
        idata = COCODataset(root=os.path.join(args.data_root, "train2014"),
                            annotation=os.path.join(args.data_root, "annotations/train2014_train_split.json"), transform_fn=transform)
    elif split == "val":
        idata = COCODataset(root=os.path.join(args.data_root, "train2014"),
                            annotation=os.path.join(args.data_root, "annotations/train2014_val_split.json"), transform_fn=transform)
    elif split == "test":
        idata = COCODataset(root=os.path.join(args.data_root, "val2014"),
                            annotation=os.path.join(args.data_root, "annotations/instances_val2014.json"), transform_fn=transform)
    save_data = torch.zeros((len(idata),)+idata[0][0].shape)
    save_mask = torch.zeros(
        (len(idata), idata[0][0].shape[1], idata[0][0].shape[2]))
    save_labels = torch.zeros((len(idata), 91))
    save_bbs = [[] for _ in range(len(idata))]

    for idx in tqdm(range(len(idata))):
        img, annotations = idata[idx]
        save_data[idx] = img
        width = annotations["width"]
        height = annotations["height"]
        wscale = 224 / width
        hscale = 224 / height
        for annotation in annotations["bboxes"]:
            save_labels[idx][annotation[0]] = 1.0
            xmin = annotation[1]
            xmax = annotation[1]+annotation[3]
            ymin = annotation[2]
            ymax = annotation[2]+annotation[4]
            new_xmin = int(min(max(xmin*wscale, 0), 223))
            new_xmax = int(min(max(xmax*wscale, 0), 223))
            new_ymin = int(min(max(ymin*hscale, 0), 223))
            new_ymax = int(min(max(ymax*hscale, 0), 223))
            save_bbs[idx].append(
                [annotation[0], new_xmin, new_ymin, new_xmax, new_ymax])

    dataset = {"data": save_data, "labels": save_labels,
                   "bbs": save_bbs, "mask": None}
    
    rm_cols = torch.tensor([0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83])
    keep_cols = torch.tensor([c for c in torch.arange(91) if c not in rm_cols])
    label_mapper = [None for _ in range(91)]
    current_offset = 0
    for col in range(91):
        if col in rm_cols:
            current_offset += 1
            continue
        label_mapper[col] = col - current_offset
    print(label_mapper)

    assert len(keep_cols) == 80
    assert dataset["labels"].shape[1] == 91
    assert dataset["labels"][:, rm_cols].sum() == 0

    dataset["labels"] = dataset["labels"][:, keep_cols]
    for img_idx in tqdm(range(len(dataset["bbs"]))):
        for bb_idx in range(len(dataset["bbs"][img_idx])):
            dataset["bbs"][img_idx][bb_idx][0] = label_mapper[dataset["bbs"][img_idx][bb_idx][0]]
    
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(dataset, os.path.join(args.save_path, split + ".pt"))


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default=".")
parser.add_argument("--split", type=str,
                    choices=["train", "val", "test"], required=True)
parser.add_argument("--save_path", type=str, default="processed/")
args = parser.parse_args()
main(args)
