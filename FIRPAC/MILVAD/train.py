import argparse
import os
from os import path
from collections import OrderedDict
from unittest import defaultTestLoader

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from .utils.loader import FeaturesLoader, FeaturesLoaderVal
from .networks.TorchUtils import TorchModel
from .networks.AnomalyDetectorModel import AnomalyDetector, custom_objective, RegularizedLoss
from .utils.callbacks import DefaultModelCallback, TensorBoardCallback
from .utils.utils import register_logger, get_torch_device
# from .evaluate import parse_args as evaluate_get_args


def train(args):
    train_args = args.train
    eval_args = args

    # Register directories
    register_logger(log_file=train_args.log_file)
    os.makedirs(train_args.exps_dir, exist_ok=True)
    models_dir = path.join(train_args.exps_dir, 'models')
    tb_dir = path.join(train_args.exps_dir, 'tensorboard')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # Optimizations
    device = get_torch_device()
    cudnn.benchmark = True  # enable cudnn tune

    if train_args.ext:
        """external data"""
        # synsets
        normal_dir = f"datasets/{train_args.trainset}/features/train/normal"
        anomaly_dir = f"datasets/{train_args.trainset}/features/train/anomaly"
        # realsets
        ex_normal_dir = train_args.normal_dir
        ex_anomaly_dir = train_args.anomaly_dir
    else:
        """synthetic data only"""
        normal_dir = f"datasets/{train_args.trainset}/features/train/normal"
        anomaly_dir = f"datasets/{train_args.trainset}/features/train/anomaly"
        ex_normal_dir = None
        ex_anomaly_dir = None
    """real data only"""
    # normal_dir = train_args.normal_dir
    # anomaly_dir = train_args.anomaly_dir
    # ex_normal_dir = None
    # ex_anomaly_dir = None

    train_loader = FeaturesLoader(
        normal_dir,
        anomaly_dir,
        bucket_size=512,
        iterations=100,
        ext=train_args.ext,  # whether use external datasets
        ex_normal_dir=ex_normal_dir,
        ex_anomaly_dir=ex_anomaly_dir)

    eval_loader = FeaturesLoaderVal(eval_args.anomaly_feature_dir, eval_args.clip_gt_dir)

    # Model
    if train_args.checkpoint is not None and path.exists(train_args.checkpoint):
        print("Resuming pretrained model {}:".format(train_args.checkpoint))
        network = AnomalyDetector(2048 * 10)
        model = TorchModel(network)
        model = TorchModel.load_model(train_args.checkpoint, model)
    else:
        network = AnomalyDetector(2048 * 10)
        model = TorchModel(network)

    model = model.to(device).train()
    # Training parameters
    """
    In the original paper:
        lr = 0.01
        epsilon = 1e-8
    """
    optimizer = torch.optim.Adadelta(model.parameters(), lr=train_args.lr_base, eps=1e-8)

    criterion = RegularizedLoss(network, custom_objective).to(device)

    # Callbacks
    tb_writer = SummaryWriter(log_dir=tb_dir)
    model.register_callback(DefaultModelCallback(visualization_dir=train_args.exps_dir))
    model.register_callback(TensorBoardCallback(tb_writer=tb_writer))

    # Training
    model.fit(
        train_iter=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        eval_iter=eval_loader,
        epochs=train_args.epochs,
        network_model_path_base=models_dir,
        #   save_every=args.save_every,
        evaluate_every=train_args.eval_every,
        args=eval_args,
    )
