"""
    Evaluation metrics of accuracy and mae.
"""
from math import sqrt

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

smooth = 1e-6


# for simultaneously person identification and localization  start
# detection-aware localization (error)
def mae_d(output, target):
    # loc_theta & sub_label
    mae_func = torch.nn.L1Loss()
    out_dim = 1
    if len(output.shape) > 1:
        out_dim = output.shape[1]
    with torch.no_grad():
        assert output.shape[0] == len(target)
        if out_dim == 40:  # ide
            mae_val_d = 180

        elif out_dim == 1:  # loc
            mae_val_d = 180

        elif out_dim == 41:  # ide&loc
            pred = torch.argmax(output[:, :40], dim=1)
            sub_label = target[:, 0]

            class_idx = (pred == sub_label)  # extract the correctly classified samples
            mae_val_d = mae_func(output[class_idx, 40] * 180.0, target[class_idx, 1] * 180.0).item()

        elif out_dim == 80:  # accdoa-il
            x_pred, y_pred = output[:, :40], output[:, 40:]
            sed_pred = torch.sqrt_(x_pred ** 2 + y_pred ** 2)
            activity_pred = torch.argmax(sed_pred, dim=1)
            # in degrees
            azimuth_pred = [torch.arctan(y_pred[i, activity_pred[i]] / x_pred[i, activity_pred[i]])*180/np.pi for i in range(len(x_pred))]

            x_gt, y_gt = target[:, :40], target[:, 40:]
            sed_gt = torch.sqrt_(x_gt ** 2 + y_gt ** 2)
            activity_gt = torch.argmax(sed_gt, dim=1)
            # in degrees
            azimuth_gt = [torch.arctan(y_gt[i, activity_gt[i]] / x_gt[i, activity_gt[i]]) * 180 / np.pi for i in range(len(x_gt))]

            # extract the correctly classified samples
            class_idx = (activity_pred == activity_gt)

            mae_val_d = mae_func(torch.asarray(azimuth_gt)[class_idx], torch.asarray(azimuth_pred)[class_idx]).cpu().numpy()

    return mae_val_d


# location-aware identification (accuracy)
def accuracy_l30(output, target):
    angular_threshold = 30
    out_dim = 1
    if len(output.shape) > 1:
        out_dim = output.shape[1]
    with torch.no_grad():
        assert output.shape[0] == len(target)
        if out_dim == 40:  # ide
            correct = 0

        elif out_dim == 1:  # loc
            correct = 0

        elif out_dim == 41:  # ide&loc
            pred = torch.argmax(output[:, :40], dim=1)
            theta_t_idx = torch.abs_(output[:, 40] * 180.0 - target[:, 1] * 180.0) < angular_threshold
            pred_l = pred[theta_t_idx]
            sub_label_l = target[:, 0][theta_t_idx]
            correct = torch.sum(pred_l == sub_label_l).item()

        elif out_dim == 80:  # accdoa-il
            x_pred, y_pred = output[:, :40], output[:, 40:]
            sed_pred = torch.sqrt_(x_pred ** 2 + y_pred ** 2)
            activity_pred = torch.argmax(sed_pred, dim=1)
            # in degrees
            azimuth_pred = [torch.arctan(y_pred[i, activity_pred[i]] / x_pred[i, activity_pred[i]]) * 180 / np.pi for i in
                            range(len(x_pred))]

            x_gt, y_gt = target[:, :40], target[:, 40:]
            sed_gt = torch.sqrt_(x_gt ** 2 + y_gt ** 2)
            activity_gt = torch.argmax(sed_gt, dim=1)
            # in degrees
            azimuth_gt = [torch.arctan(y_gt[i, activity_gt[i]] / x_gt[i, activity_gt[i]]) * 180 / np.pi for i in
                          range(len(x_gt))]

            theta_t_idx = torch.abs_(torch.asarray(azimuth_gt) - torch.asarray(azimuth_pred)) < angular_threshold
            pred_l = activity_pred[theta_t_idx]
            sub_label_l = activity_gt[theta_t_idx]
            correct = torch.sum(pred_l == sub_label_l).item()

    return correct / len(target)


# location-aware identification (accuracy)
def accuracy_l(output, target):
    angular_threshold = 20
    out_dim = 1
    if len(output.shape) > 1:
        out_dim = output.shape[1]
    with torch.no_grad():
        assert output.shape[0] == len(target)
        if out_dim == 40:  # ide
            correct = 0

        elif out_dim == 1:  # loc
            correct = 0

        elif out_dim == 41:  # ide&loc
            pred = torch.argmax(output[:, :40], dim=1)
            theta_t_idx = torch.abs_(output[:, 40] * 180.0 - target[:, 1] * 180.0) < angular_threshold
            pred_l = pred[theta_t_idx]
            sub_label_l = target[:, 0][theta_t_idx]
            correct = torch.sum(pred_l == sub_label_l).item()

        elif out_dim == 80:  # accdoa-il
            x_pred, y_pred = output[:, :40], output[:, 40:]
            sed_pred = torch.sqrt_(x_pred ** 2 + y_pred ** 2)
            activity_pred = torch.argmax(sed_pred, dim=1)
            # in degrees
            azimuth_pred = [torch.arctan(y_pred[i, activity_pred[i]] / x_pred[i, activity_pred[i]]) * 180 / np.pi for i in
                            range(len(x_pred))]

            x_gt, y_gt = target[:, :40], target[:, 40:]
            sed_gt = torch.sqrt_(x_gt ** 2 + y_gt ** 2)
            activity_gt = torch.argmax(sed_gt, dim=1)
            # in degrees
            azimuth_gt = [torch.arctan(y_gt[i, activity_gt[i]] / x_gt[i, activity_gt[i]]) * 180 / np.pi for i in
                          range(len(x_gt))]

            theta_t_idx = torch.abs_(torch.asarray(azimuth_gt) - torch.asarray(azimuth_pred)) < angular_threshold
            pred_l = activity_pred[theta_t_idx]
            sub_label_l = activity_gt[theta_t_idx]
            correct = torch.sum(pred_l == sub_label_l).item()

    return correct / len(target)

# for simultaneously person identification and localization  end


def accuracy(output, target):
    out_dim = 1
    if len(output.shape) > 1:
        out_dim = output.shape[1]
    with (torch.no_grad()):
        assert output.shape[0] == len(target)
        if out_dim == 40:  # ide
            pred = torch.argmax(output, dim=1)
            correct = torch.sum(pred == target).item()

        elif out_dim == 1:  # loc
            correct = 0

        elif out_dim == 41:  # ide&loc
            pred = output[:, :40]
            activity_pred = torch.argmax(pred, dim=1)
            correct = torch.sum(activity_pred == target[:, 0]).item()

        elif out_dim == 80:  # accdoa-il
            x_pred, y_pred = output[:, :40], output[:, 40:]
            sed_pred = torch.sqrt_(x_pred ** 2 + y_pred ** 2)
            activity_pred = torch.argmax(sed_pred, dim=1)

            x_gt, y_gt = target[:, :40], target[:, 40:]
            sed_gt = torch.sqrt_(x_gt ** 2 + y_gt ** 2)
            activity_gt = torch.argmax(sed_gt, dim=1)

            correct = torch.sum(activity_pred == activity_gt).item()

    return correct / len(target)


def mae(output, target):
    mae_func = torch.nn.L1Loss()
    out_dim = 1
    if len(output.shape) > 1:
        out_dim = output.shape[1]
    with torch.no_grad():
        assert output.shape[0] == len(target)
        if out_dim == 40:  # ide
            mae_val = 180

        elif out_dim == 1:  # loc
            mae_val = mae_func(output*180.0, target*180.0).item()

        elif out_dim == 41:  # ide&loc
            pred = output[:, 40]
            # rad convert degree
            mae_val = mae_func(pred*180.0, target[:, 1]*180.0).item()

        elif out_dim == 80:  # accdoa-il
            x_pred, y_pred = output[:, :40], output[:, 40:]
            sed_pred = torch.sqrt_(x_pred ** 2 + y_pred ** 2)
            activity_pred = torch.argmax(sed_pred, dim=1)
            # in degrees
            azimuth_pred = [torch.arctan(y_pred[i, activity_pred[i]] / x_pred[i, activity_pred[i]]) * 180 / np.pi for i in
                            range(len(x_pred))]

            x_gt, y_gt = target[:, :40], target[:, 40:]
            sed_gt = torch.sqrt_(x_gt ** 2 + y_gt ** 2)
            activity_gt = torch.argmax(sed_gt, dim=1)
            # in degrees
            azimuth_gt = [torch.arctan(y_gt[i, activity_gt[i]] / x_gt[i, activity_gt[i]]) * 180 / np.pi for i in
                          range(len(x_gt))]

            mae_val = mae_func(torch.asarray(azimuth_gt), torch.asarray(azimuth_pred)).cpu().numpy()

    return mae_val


class MetricTracker:
    def __init__(self, keys_iter: list, keys_epoch: list, writer=None):
        self.writer = writer
        self.metrics_iter = pd.DataFrame(index=keys_iter, columns=['current', 'sum', 'square_sum', 'counts',
                                                                   'mean', 'square_avg', 'std'])
        self.metrics_epoch = pd.DataFrame(index=keys_epoch, columns=['mean'])
        self.reset()

    def reset(self):
        for col in self.metrics_iter.columns:
            self.metrics_iter[col].values[:] = 0

    def iter_update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self.metrics_iter.at[key, 'current'] = value
        self.metrics_iter.at[key, 'sum'] += value * n
        self.metrics_iter.at[key, 'square_sum'] += value * value * n
        self.metrics_iter.at[key, 'counts'] += n

    def epoch_update(self, key, value):
        self.metrics_epoch.at[key, 'mean'] = value

    def current(self):
        return dict(self.metrics_iter['current'])

    def avg(self):
        for key, row in self.metrics_iter.iterrows():
            self.metrics_iter.at[key, 'mean'] = row['sum'] / row['counts']
            self.metrics_iter.at[key, 'square_avg'] = row['square_sum'] / row['counts']

    def std(self):
        for key, row in self.metrics_iter.iterrows():
            self.metrics_iter.at[key, 'std'] = sqrt(row['square_avg'] - row['mean'] ** 2 + smooth)

    def result(self):
        self.avg()
        self.std()
        iter_result = self.metrics_iter[['mean', 'std']]
        epoch_result = self.metrics_epoch

        return pd.concat([iter_result, epoch_result])


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def binary_accuracy(output, target):
    with torch.no_grad():
        correct = 0
        correct += torch.sum(torch.abs(output - target) < 0.5).item()
    return correct / len(target)


def AUROC(output, target):
    with torch.no_grad():
        value = roc_auc_score(target.cpu().numpy(), output.cpu().numpy())
    return value


def AUPRC(output, target):
    with torch.no_grad():
        value = average_precision_score(target.cpu().numpy(), output.cpu().numpy())
    return value


def noise_RoCAUC(output, target):
    with torch.no_grad():
        value = roc_auc_score(target.cpu().numpy()[:, 0], output.cpu().numpy()[:, 0])
    return value


def noise_AP(output, target):
    with torch.no_grad():
        value = average_precision_score(target.cpu().numpy()[:, 0], output.cpu().numpy()[:, 0])
    return value


def mean_iou_score(output, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        pred = pred.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        mean_iou = 0
        for i in range(6):
            tp_fp = np.sum(pred == i)
            tp_fn = np.sum(labels == i)
            tp = np.sum((pred == i) * (labels == i))
            iou = (tp + smooth) / (tp_fp + tp_fn - tp + smooth)
            mean_iou += iou / 6

    return mean_iou
