import torch
import torchvision
import argparse
import os
from tqdm import tqdm

"""
Source: https://github.com/stevenstalder/NN-Explainer
"""
def get_target_dictionary(include_background_class):
    if include_background_class:
        target_dict = {'background': 0, 'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7,
                       'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
                       'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}
    else:
        target_dict = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6,
                       'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                       'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

    return target_dict


def main(args):
    split = args.split
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(
        size=(224, 224)), torchvision.transforms.ToTensor()])
    idata = torchvision.datasets.VOCDetection(
            root=args.data_root, year="2007", download=True, image_set=split, transform=transform)
    
    save_data = torch.zeros((len(idata),)+idata[0][0].shape)
    save_labels = torch.zeros((len(idata), 20))
    save_bbs = [[] for _ in range(len(idata))]

    for idx in tqdm(range(len(idata))):
        img, annotations = idata[idx]
        save_data[idx] = img
        target_dict = get_target_dictionary(False)
        objects = annotations['annotation']['object']
        size = annotations['annotation']['size']
        width = int(size['width'])
        height = int(size['height'])
        wscale = 224 / width
        hscale = 224 / height
        object_names = [item['name'] for item in objects]
        for name in object_names:
            index = target_dict[name]
            save_labels[idx][index] = 1.0
        object_bndboxes = [item['bndbox'] for item in objects]
        for name, bndbox in zip(object_names, object_bndboxes):
            index = target_dict[name]
            xmin = int(bndbox['xmin'])
            xmax = int(bndbox['xmax'])
            ymin = int(bndbox['ymin'])
            ymax = int(bndbox['ymax'])
            new_xmin = int(min(max(xmin*wscale, 0), 223))
            new_xmax = int(min(max(xmax*wscale, 0), 223))
            new_ymin = int(min(max(ymin*hscale, 0), 223))
            new_ymax = int(min(max(ymax*hscale, 0), 223))
            save_bbs[idx].append(
                [index, new_xmin, new_ymin, new_xmax, new_ymax])

    dataset = {"data": save_data, "labels": save_labels,
                   "bbs": save_bbs, "mask": None}
    
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(dataset, os.path.join(args.save_path, split + ".pt"))


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default=".")
parser.add_argument("--split", type=str,
                    choices=["train", "val", "test", "trainval"], required=True)
parser.add_argument("--save_path", type=str, default="processed/")
args = parser.parse_args()
main(args)
