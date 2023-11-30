import os
import argparse
import collections
from operator import itemgetter
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from base import Cross_Valid
from logger import get_logger
import models.loss as module_loss
import models.metric as module_metric
from models.metric import MetricTracker
from parse_config import ConfigParser
from utils import ensure_dir, prepare_device, get_by_path, msg_box
from sklearn.metrics import log_loss
import numexpr as ne

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['NUMEXPR_MAX_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):

    logger = get_logger('test')
    test_msg = msg_box("TEST: "+config.run_args.task)
    logger.debug(test_msg)

    # datasets
    test_datasets = dict()
    keys = ['datasets', 'test']
    for name in get_by_path(config, keys):
        test_datasets[name] = config.init_obj([*keys, name], 'data_loaders')

    # data_loaders
    test_data_loaders = dict()
    keys = ['data_loaders', 'test']
    for name in get_by_path(config, keys):
        dataset = test_datasets[name]
        test_data_loaders[name] = config.init_obj([*keys, name], 'data_loaders', dataset)

    # prepare model for testing
    device, device_ids = prepare_device(config['n_gpu'])

    # models
    resume = config.resume
    logger.info(f"Loading model: {resume} ...")
    checkpoint = torch.load(resume)
    models = dict()
    logger_model = get_logger('model', verbosity=0)
    for name in config['models']:
        model = config.init_obj(['models', name], 'models')
        logger_model.info(model)
        state_dict = checkpoint['models'][name]
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models[name] = model

    losses = dict()
    for name in config['losses']:
        kwargs = {}
        losses[name] = config.init_obj(['losses', name], module_loss, **kwargs)

    # metrics
    metrics_iter = [getattr(module_metric, met) for met in config['metrics']['per_iteration']]
    metrics_epoch = [getattr(module_metric, met) for met in config['metrics']['per_epoch']]
    keys_loss = ['loss']
    keys_iter = [m.__name__ for m in metrics_iter]
    keys_epoch = [m.__name__ for m in metrics_epoch]
    test_metrics = MetricTracker(keys_loss + keys_iter, keys_epoch)

    task = config.run_args.task.split('_')[0]

    with torch.no_grad():
        print("testing...")
        model = models['model']
        testloader = test_data_loaders['data']
        if len(metrics_epoch) > 0:  # true
            outputs = torch.FloatTensor().to(device)
            targets = torch.FloatTensor().to(device)
        # for batch_idx, (data, loc_theta, target) in tqdm(enumerate(testloader), total=len(testloader)):
        for batch_idx, (data, target) in tqdm(enumerate(testloader), total=len(testloader)):
            if isinstance(data, dict):
                data = {k: v.to(device).float() for k, v in data.items()}
            else:
                data = data.to(device).float()

            # sub_label & loc_theta
            target = target.to(device).float()

            output = model(data)
            if len(metrics_epoch) > 0:  # true
                outputs = torch.cat((outputs, output))
                targets = torch.cat((targets, target))

            # save sample images, or do something with output here

            # computing loss, metrics on test set
            if task == 'ide':
                loss = losses['ide_loss'](output, target.type(torch.int64))

            elif task == 'loc':
                loss = losses['loc_loss'](output, target)

            elif task == 'ideloc':
                ide_loss = losses['ide_loss'](output[:, :40], target[:, 0].type(torch.int64))
                loc_loss = losses['loc_loss'](output[:, 40], target[:, 1])
                loss = ide_loss + loc_loss

            elif task == 'accil':
                loss = losses['accdoa_loss'](output, target)

            test_metrics.iter_update('loss', loss.item())
            for met in metrics_iter:
                test_metrics.iter_update(met.__name__, met(output, target))

        for met in metrics_epoch:
            test_metrics.epoch_update(met.__name__, met(outputs, targets))

    print(outputs.cpu().numpy().shape)

    # saving prediction and targets
    save_dir = config.save_dir["metric"]
    np.savez(os.path.join(save_dir, 'pred_target'), pred=outputs.cpu().numpy(), target=targets.cpu().numpy())


    # for classification ===============================================================================================
    # outputs = nn.Softmax(dim=1)(outputs)
    #
    # preds = torch.argmax(outputs, dim=1)

    test_log = test_metrics.result()
    logger.info(test_log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='testing')
    run_args = args.add_argument_group('run_args')
    run_args.add_argument('-c', '--config', default="configs/afpild_spec_gcc_fusion.json", type=str)
    run_args.add_argument('-r', '--resume', default='saved/AFPILD-CRNN/accil_ori_cloth/model/model_best.pth', type=str)
    run_args.add_argument('-d', '--device', default=None, type=str)
    run_args.add_argument('--mode', default='test', type=str)
    run_args.add_argument('--run_id', default='Fusion Model', type=str)
    run_args.add_argument('--log_name', default=None, type=str)

    'ide_ori_{rd/cloth/shoe},'
    'loc_ori_{rd/cloth/shoe}, '
    'ideloc_ori_{rd/cloth/shoe},'
    'accil_ori_{rd/cloth/shoe}, accil_ana_{rd/cloth/shoe}'
    run_args.add_argument('-t', '--task', default='accil_ori_cloth', type=str)

    # custom cli options to modify configuration from default values given in json file.
    mod_args = args.add_argument_group('mod_args')
    CustomArgs = collections.namedtuple('CustomArgs', "flags default type target")
    options = [

    ]
    for opt in options:
        mod_args.add_argument(*opt.flags, default=opt.default, type=opt.type)

    # additional arguments for testing
    test_args = args.add_argument_group('test_args')
    test_args.add_argument('--output_path', default=None, type=str)

    cfg = ConfigParser.from_args(args, options)
    main(cfg)
