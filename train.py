import torch
import os
import argparse
import torchvision
from tqdm import tqdm
import datasets
import argparse
import torch.utils.tensorboard
import utils
import copy
import losses
import metrics
import bcos.models
import model_activators
import attribution_methods
import hubconf
import bcos
import bcos.modules
import bcos.data
import fixup_resnet


def eval_model(model, attributor, loader, num_batches, num_classes, loss_fn, writer=None, epoch=None):
    model.eval()
    f1_metric = metrics.MultiLabelMetrics(
        num_classes=num_classes, threshold=0.0)
    bb_metric = metrics.BoundingBoxEnergyMultiple()
    iou_metric = metrics.BoundingBoxIoUMultiple()
    total_loss = 0
    for batch_idx, (test_X, test_y, test_bbs) in enumerate(tqdm(loader)):
        test_X.requires_grad = True
        test_X = test_X.cuda()
        test_y = test_y.cuda()
        logits, features = model(test_X)
        loss = loss_fn(logits, test_y).detach()
        total_loss += loss
        f1_metric.update(logits, test_y)

        if attributor:
            for img_idx in range(len(test_X)):
                class_target = torch.where(test_y[img_idx] == 1)[0]
                for pred_idx, pred in enumerate(class_target):
                    attributions = attributor(
                        features, logits, pred, img_idx).detach().squeeze(0).squeeze(0)
                    bb_list = utils.filter_bbs(test_bbs[img_idx], pred)
                    bb_metric.update(attributions, bb_list)
                    iou_metric.update(attributions, bb_list)

    metric_vals = f1_metric.compute()
    if attributor:
        bb_metric_vals = bb_metric.compute()
        iou_metric_vals = iou_metric.compute()
        metric_vals["BB-Loc"] = bb_metric_vals
        metric_vals["BB-IoU"] = iou_metric_vals
    metric_vals["Average-Loss"] = total_loss.item()/num_batches        
    print(f"Validation Metrics: {metric_vals}")
    model.train()
    if writer is not None:
        writer.add_scalar("val_loss", total_loss.item()/num_batches, epoch)
        writer.add_scalar("accuracy", metric_vals["Accuracy"], epoch)
        writer.add_scalar("precision", metric_vals["Precision"], epoch)
        writer.add_scalar("recall", metric_vals["Recall"], epoch)
        writer.add_scalar("fscore", metric_vals["F-Score"], epoch)
        if attributor:
            writer.add_scalar("bbloc", metric_vals["BB-Loc"], epoch)
            writer.add_scalar("bbiou", metric_vals["BB-IoU"], epoch)
    return metric_vals


def main(args):
    utils.set_seed(args.seed)

    num_classes_dict = {"VOC2007": 20, "COCO2014":  80}
    num_classes = num_classes_dict[args.dataset]

    is_bcos = (args.model_backbone == "bcos")
    is_xdnn = (args.model_backbone == "xdnn")
    is_vanilla = (args.model_backbone == "vanilla")
        

    if is_bcos:
        model = hubconf.resnet50(pretrained=True)
        model[0].fc = bcos.modules.bcosconv2d.BcosConv2d(
                    in_channels=model[0].fc.in_channels, out_channels=num_classes)
        layer_dict = {"Input": None, "Mid1": 3,
                      "Mid2": 4, "Mid3": 5, "Final": 6}
    elif is_xdnn:
        model = fixup_resnet.xfixup_resnet50()
        imagenet_checkpoint = torch.load(os.path.join("weights/xdnn/xfixup_resnet50_model_best.pth.tar"))
        imagenet_state_dict = utils.remove_module(
            imagenet_checkpoint["state_dict"])
        model.load_state_dict(imagenet_state_dict)
        model.fc = torch.nn.Linear(
            in_features=model.fc.in_features, out_features=num_classes)
        layer_dict = {"Input": None, "Mid1": 3,
                      "Mid2": 4, "Mid3": 5, "Final": 6}
    elif is_vanilla:
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(
                in_features=model.fc.in_features, out_features=num_classes)
        layer_dict = {"Input": None, "Mid1": 4,
                      "Mid2": 5, "Mid3": 6, "Final": 7}
    else:
        raise NotImplementedError

    layer_idx = layer_dict[args.layer]

    if args.model_path is not None:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint["model"])

    model = model.cuda()
    model.train()

    orig_name = os.path.basename(
        args.model_path) if args.model_path else str(None)

    model_prefix = args.model_backbone

    optimize_explanation_str = "finetunedobjloc" if args.optimize_explanations else "standard"
    optimize_explanation_str += "pareto" if args.pareto else ""
    optimize_explanation_str += "limited" if args.annotated_fraction < 1.0 else ""
    optimize_explanation_str += "dilated" if args.box_dilation_percentage > 0 else ""

    out_name = model_prefix + "_" + optimize_explanation_str + "_attr" + str(args.attribution_method) + "_locloss" + str(args.localization_loss_fn) + "_orig" + orig_name + "_resnet50" + "_lr" + str(
        args.learning_rate) + "_sll" + str(args.localization_loss_lambda) + "_layer" + str(args.layer)
    if args.annotated_fraction < 1.0:
        out_name += f"limited{args.annotated_fraction}"
    if args.box_dilation_percentage > 0:
        out_name += f"_dilation{args.box_dilation_percentage}"

    save_path = os.path.join(args.save_path, args.dataset, out_name)
    os.makedirs(save_path, exist_ok=True)

    if args.log_path is not None:
        writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=os.path.join(args.log_path, args.dataset, out_name))
    else:
        writer = None

    if is_bcos:
        transformer = bcos.data.transforms.AddInverse(dim=0)
    else:
        transformer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])

    root = os.path.join(args.data_path, args.dataset, "processed")
    train_data = datasets.VOCDetectParsed(
        root=root, image_set="train", transform=transformer, annotated_fraction=args.annotated_fraction)
    val_data = datasets.VOCDetectParsed(
        root=root, image_set="val", transform=transformer)
    test_data = datasets.VOCDetectParsed(
        root=root, image_set="test", transform=transformer)

    print(f"Train data size: {len(train_data)}")
    annotation_count = 0
    total_count = 0
    for idx in range(len(train_data)):
        if train_data[idx][2] is not None:
            annotation_count += 1
        total_count += 1
    print(f"Annotated: {annotation_count}, Total: {total_count}")

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=4, collate_fn=datasets.VOCDetectParsed.collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=4, collate_fn=datasets.VOCDetectParsed.collate_fn)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=4, collate_fn=datasets.VOCDetectParsed.collate_fn)

    num_train_batches = len(train_data) / args.train_batch_size
    num_val_batches = len(val_data) / args.eval_batch_size
    num_test_batches = len(test_data) / args.eval_batch_size

    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_loc = losses.get_localization_loss(
        args.localization_loss_fn) if args.localization_loss_fn else None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    f1_tracker = utils.BestMetricTracker("F-Score")
    
    model_activator = model_activators.ResNetModelActivator(
        model=model, layer=layer_idx, is_bcos=is_bcos)
    
    if args.attribution_method:
        interpolate = True if layer_idx is not None else False
        attributor = attribution_methods.get_attributor(
                model, args.attribution_method, loss_loc.only_positive, loss_loc.binarize, interpolate, (224, 224), batch_mode=True)       
        eval_attributor = attribution_methods.get_attributor(
            model, args.attribution_method, loss_loc.only_positive, loss_loc.binarize, interpolate, (224, 224), batch_mode=False)
    else:
        attributor = None
        eval_attributor = None

    if args.pareto:
        pareto_front_tracker = utils.ParetoFrontModels()

    for e in tqdm(range(args.total_epochs)):
        total_loss = 0
        total_class_loss = 0
        total_localization_loss = 0

        for batch_idx, (train_X, train_y, train_bbs) in enumerate(tqdm(train_loader)):
            batch_loss = 0
            localization_loss = 0
            optimizer.zero_grad()
            train_X.requires_grad = True
            train_X = train_X.cuda()
            train_y = train_y.cuda()
            logits, features = model_activator(train_X)
            loss = loss_fn(logits, train_y)
            batch_loss += loss
            total_class_loss += loss.detach()

            if args.optimize_explanations:
                gt_classes = utils.get_random_optimization_targets(train_y)
                attributions = attributor(
                    features, logits, classes=gt_classes).squeeze(1)
                for img_idx in range(len(train_X)):
                    if train_bbs[img_idx] is None:
                        continue
                    bb_list = utils.filter_bbs(
                        train_bbs[img_idx], gt_classes[img_idx])
                    if args.box_dilation_percentage > 0:
                        bb_list = utils.enlarge_bb(
                            bb_list, percentage=args.box_dilation_percentage)
                    localization_loss += loss_loc(attributions[img_idx], bb_list)
                batch_loss += args.localization_loss_lambda*localization_loss
                if torch.is_tensor(localization_loss):
                    total_localization_loss += localization_loss.detach()
                else:
                    total_localization_loss += localization_loss
               
            batch_loss.backward()
            total_loss += batch_loss.detach()
            optimizer.step()

        print(f"Epoch: {e}, Average Loss: {total_loss / num_train_batches}")

        if writer:
            writer.add_scalar("train_loss", total_loss, e+1)
            writer.add_scalar("class_loss", total_class_loss, e+1)
            writer.add_scalar("localization_loss", total_localization_loss, e+1)
        if (e+1) % args.evaluation_frequency == 0:
            metric_vals = eval_model(model_activator, eval_attributor, val_loader,
                                     num_val_batches, num_classes, loss_fn, writer, e)
            if args.pareto:
                pareto_front_tracker.update(model, metric_vals, e)
            best_fscore, _, _, _ = f1_tracker.get_best()
            if (best_fscore is not None) and (best_fscore < args.min_fscore):
                print(
                    f'F-Score below threshold, actual: {metric_vals["F-Score"]}, threshold: {args.min_fscore}')
                metric_vals.update(
                    {"model": None, "epochs": e+1} | vars(args))
                metric_vals.update({"BelowThresh": True})
                torch.save(metric_vals, os.path.join(
                    save_path, f"model_checkpoint_stopped_{e+1}.pt"))
                if args.pareto:
                    pareto_front_tracker.save_pareto_front(save_path)
                return
            f1_tracker.update(metric_vals, model, e)

    if args.pareto:
        pareto_front_tracker.save_pareto_front(save_path)

    final_metric_vals = metric_vals
    final_metric_vals = utils.update_val_metrics(final_metric_vals)
    final_metrics = eval_model(
        model_activator, eval_attributor, test_loader, num_test_batches, num_classes, loss_fn)
    final_state_dict = copy.deepcopy(model.state_dict())
    final_metrics.update(final_metric_vals)
    final_metrics.update(
        {"model": final_state_dict, "epochs": e+1} | vars(args))

    f1_best_score, f1_best_model_dict, f1_best_epoch, f1_best_metric_vals = f1_tracker.get_best()
    f1_best_metric_vals = utils.update_val_metrics(f1_best_metric_vals)
    model.load_state_dict(f1_best_model_dict)
    f1_best_metrics = eval_model(model_activator, eval_attributor, test_loader,
                                 num_test_batches, num_classes, loss_fn)
    f1_best_metrics.update(f1_best_metric_vals)
    f1_best_metrics.update(
        {"model": f1_best_model_dict, "epochs": f1_best_epoch+1} | vars(args))

    torch.save(final_metrics, os.path.join(
        save_path, f"model_checkpoint_final_{e+1}.pt"))
    torch.save(f1_best_metrics, os.path.join(
        save_path, f"model_checkpoint_f1_best.pt"))


parser = argparse.ArgumentParser()
parser.add_argument("--model_backbone", type=str, choices=["bcos", "xdnn", "vanilla"], required=True, help="Model backbone to train.")
parser.add_argument("--model_path", type=str, default=None, help="Path to checkpoint to fine tune from. When None, a model is trained starting from ImageNet pre-trained weights.")
parser.add_argument("--data_path", type=str, default="datasets/", help="Path to datasets.")
parser.add_argument("--total_epochs", type=int, default=100, help="Number of epochs to train for.")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate to use.")
parser.add_argument("--log_path", type=str, default=None, help="Path to save TensorBoard logs.")
parser.add_argument("--save_path", type=str, default="checkpoints/", help="Path to save trained models.")
parser.add_argument("--seed", type=int, default=0, help="Random seed to use.")
parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size to use for training.")
parser.add_argument("--dataset", type=str, required=True,
                    choices=["VOC2007", "COCO2014"], help="Dataset to train on.")
parser.add_argument("--localization_loss_lambda", type=float, default=1.0, help="Lambda to use to weight localization loss.")
parser.add_argument("--layer", type=str, default="Input",
                    choices=["Input", "Final", "Mid1", "Mid2", "Mid3"], help="Layer of the model to compute and optimize attributions on.")
parser.add_argument("--localization_loss_fn", type=str, default=None,
                    choices=["Energy", "L1", "RRR", "PPCE"], help="Localization loss function to use.")
parser.add_argument("--attribution_method", type=str, default=None,
                    choices=["BCos", "GradCam", "IxG"], help="Attribution method to use for optimization.")
parser.add_argument("--optimize_explanations",
                    action="store_true", default=False, help="Flag for optimizing attributions. When False, a model is trained just using the classification loss.")
parser.add_argument("--min_fscore", type=float, default=-1, help="Minimum F-Score the best model so far must have to continue training. If the best F-Score drops below this threshold, stops training early.")
parser.add_argument("--pareto", action="store_true", default=False, help="Flag to save Pareto front of models based on F-Score, EPG Score, and IoU Score.")
parser.add_argument("--annotated_fraction", type=float, default=1.0, help="Fraction of training dataset from which bounding box annotations are to be used.")
parser.add_argument("--evaluation_frequency", type=int, default=1, help="Frequency (number of epochs) at which to evaluate the current model.")
parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size to use for evaluation.")
parser.add_argument("--box_dilation_percentage", type=float, default=0, help="Fraction of dilation to use for bounding boxes when training.")
args = parser.parse_args()
main(args)
