import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

import torch.nn.functional as F
def inf_norm(model_t, model_s, eps=1e-3):
    
    device = next(model_t.parameters()).device
    target = torch.tensor(eps).to(device)

    model_s_dict = {n:p for n, p in model_s.named_parameters()}
    model_t_dict = {n:p for n, p in model_t.named_parameters()}

    pair_num, norm_error = 0, torch.tensor(0.0).to(device)
    for name_t, param_t in model_t_dict.items():
        if 'weight' in name_t: # only compress linear layer with weight
            name_s1 = name_t.replace('.weight', '.0.weight')
            name_s2 = name_t.replace('.weight', '.1.weight')
            
            # some params are not compressed
            if name_s1 in model_s_dict and name_s2 in model_s_dict: 
                pair_num += 1
                param1 = model_s_dict[name_s1]
                param2 = model_s_dict[name_s2]
                rebuild = torch.mm(param2, param1)

                error = F.mse_loss(torch.norm(rebuild - param_t, float('inf')), target) 
                norm_error += error

    return norm_error / pair_num


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):

    if args.distillation:
        model_t, model = model
        model_t.eval()

    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
                    
        with torch.cuda.amp.autocast():
            if args.distillation:
                feature_s, logits_s = model(samples, output_feature=True)
                outputs = logits_s
                with torch.no_grad():
                    feature_t, logits_t = model_t(samples, output_feature=True)

                L_fm = 0.
                for feat_s, feat_t in zip(feature_s, feature_t):
                    L_fm += torch.nn.functional.mse_loss(feat_s, feat_t.detach())                    
                L_fm = L_fm / len(feature_s)

                L_sr = torch.nn.functional.mse_loss(logits_s, logits_t)

            if args.inf_norm_weight > 0:
                loss_inf_norm = inf_norm(model_t, model, args.inf_norm_eps)

            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

            if args.distillation:
                L_fm = L_fm * args.kd_alpha
                L_sr = L_sr * args.kd_beta
                loss = loss + L_fm + L_sr

            if args.inf_norm_weight > 0:
                loss += loss_inf_norm * args.inf_norm_weight 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    if isinstance(model, list):
        model_t, model = model

    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_feature(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    if isinstance(model, list):
        model_t, model = model

    model.eval()

    feature_list, label_list = [], []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            features, output = model(images, output_feature=True)
            loss = criterion(output, target)

            feature_list.append(features[-1][:, 0, :].cpu())
            label_list.append(target.cpu())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    

    features = torch.cat(feature_list, dim=0)
    labels = torch.cat(label_list)
    torch.save(features, "features_direct_lowrank.pth")
    torch.save(labels, "labels_direct_lowrank.pth")

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
