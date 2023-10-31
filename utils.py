import torch
import random
import numpy as np
import copy
from collections import OrderedDict
import os


def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def filter_bbs(bb_coordinates, gt):
    bb_list = []
    for bb in bb_coordinates:
        if bb[0] == gt:
            bb_list.append(bb[1:])
    return bb_list


class BestMetricTracker:

    def __init__(self, metric_name):
        super().__init__()
        self.metric_name = metric_name
        self.best_model_dict = None
        self.best_epoch = None
        self.best_metrics = None
        self.initialized = False

    def update_values(self, metric_dict, model, epoch):
        self.best_model_dict = copy.deepcopy(model.state_dict())
        self.best_metrics = copy.deepcopy(metric_dict)
        self.best_epoch = epoch

    def update(self, metric_dict, model, epoch):
        if not self.initialized:
            self.update_values(metric_dict, model, epoch)
            self.initialized = True
        elif self.best_metrics[self.metric_name] < metric_dict[self.metric_name]:
            self.update_values(metric_dict, model, epoch)

    def get_best(self):
        if not self.initialized:
            return None, None, None, None
        return self.best_metrics[self.metric_name], self.best_model_dict, self.best_epoch, self.best_metrics



def get_random_optimization_targets(targets):
    probabilities = targets/targets.sum(dim=1, keepdim=True).detach()
    return probabilities.multinomial(num_samples=1).squeeze(1)


class ParetoFrontModels:

    def __init__(self, bin_width=0.005):
        super().__init__()
        self.bin_width = bin_width
        self.pareto_checkpoints = []
        self.pareto_costs = []

    def update(self, model, metric_dict, epoch):
        metric_vals = copy.deepcopy(metric_dict)
        state_dict = copy.deepcopy(model.state_dict())
        metric_vals.update({"model": state_dict, "epochs": epoch+1})
        self.pareto_checkpoints.append(metric_vals)
        self.pareto_costs.append(
            [metric_vals["F-Score"], metric_vals["BB-Loc"], metric_vals["BB-IoU"]])
        efficient_indices = self.is_pareto_efficient(
            -np.round(np.array(self.pareto_costs) / self.bin_width, 0)*self.bin_width, return_mask=False)
        self.pareto_checkpoints = [
            self.pareto_checkpoints[idx] for idx in efficient_indices]
        self.pareto_costs = [self.pareto_costs[idx]
                             for idx in efficient_indices]
        print(f"Current Pareto Front Size: {len(self.pareto_checkpoints)}")
        pareto_str = ""
        for idx, cost in enumerate(self.pareto_costs):
            pareto_str += f"({cost[0]:.4f},{cost[1]:.4f},{cost[2]:.4f},{self.pareto_checkpoints[idx]['epochs']})"
        print(f"Pareto Costs: {pareto_str}")

    def get_pareto_front(self):
        return self.pareto_checkpoints, self.pareto_costs

    def save_pareto_front(self, save_path):
        augmented_path = os.path.join(save_path, "pareto_front")
        os.makedirs(augmented_path, exist_ok=True)
        for idx in range(len(self.pareto_checkpoints)):
            f_score = self.pareto_checkpoints[idx]["F-Score"]
            bb_score = self.pareto_checkpoints[idx]["BB-Loc"]
            iou_score = self.pareto_checkpoints[idx]["BB-IoU"]
            epoch = self.pareto_checkpoints[idx]["epochs"]
            torch.save(self.pareto_checkpoints[idx], os.path.join(
                augmented_path, f"model_checkpoint_pareto_{f_score:.4f}_{bb_score:.4f}_{iou_score:.4f}_{epoch}.pt"))

    def is_pareto_efficient(self, costs, return_mask=True):
        """
        Find the pareto-efficient points
        : param costs: An(n_points, n_costs) array
        : param return_mask: True to return a mask
        : return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an(n_points, ) boolean array
            Otherwise it will be a(n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(
                costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            # Remove dominated points
            is_efficient = is_efficient[nondominated_point_mask]
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(
                nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient


def enlarge_bb(bb_list, percentage=0):
    en_bb_list = []
    for bb_coord in bb_list:
        xmin, ymin, xmax, ymax = bb_coord
        width = xmax - xmin
        height = ymax - ymin
        w_margin = int(percentage * width)
        h_margin = int(percentage * height)
        new_xmin = max(0, xmin-w_margin)
        new_xmax = min(223, xmax+w_margin)
        new_ymin = max(0, ymin-h_margin)
        new_ymax = min(223, ymax+h_margin)
        en_bb_list.append([new_xmin, new_ymin, new_xmax, new_ymax])
    return en_bb_list


def update_val_metrics(metric_vals):
    metric_vals["Val-Accuracy"] = metric_vals.pop("Accuracy")
    metric_vals["Val-Precision"] = metric_vals.pop("Precision")
    metric_vals["Val-Recall"] = metric_vals.pop("Recall")
    metric_vals["Val-F-Score"] = metric_vals.pop("F-Score")
    metric_vals["Val-Average-Loss"] = metric_vals.pop("Average-Loss")
    if "BB-Loc" in metric_vals:
        metric_vals["Val-BB-Loc"] = metric_vals.pop("BB-Loc")
        metric_vals["Val-BB-IoU"] = metric_vals.pop("BB-IoU")
    return metric_vals