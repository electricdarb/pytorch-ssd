#
# train an SSD model on Pascal VOC or Open Images datasets
#
import os
import itertools
import torch
from filelock import FileLock

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd

from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

    return running_loss / num, running_regression_loss / num, running_classification_loss / num

def get_data_loaders():
    dataset_paths = ['/mnt/c/Users/14135/Desktop/pytorch-ssd/data']
    datasets = []

    config = mobilenetv1_ssd_config

    with FileLock(os.path.expanduser("~/data.lock")):

        for dataset_path in dataset_paths:

            train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
            target_transform = MatchPrior(config.priors, config.center_variance,config.size_variance, 0.5)

            test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

            dataset = VOCDataset(dataset_path, transform = train_transform, target_transform = target_transform)

            datasets.append(dataset)

        # create training dataset
        train_dataset = ConcatDataset(datasets)

        train_loader = DataLoader(train_dataset, 1, num_workers = 2, shuffle=True)

        # create training dataset
        train_dataset = ConcatDataset(datasets)

        train_loader = DataLoader(train_dataset, 1,
                                num_workers = 2, shuffle = True)
        
        val_dataset = VOCDataset(dataset_path, transform=test_transform, target_transform=target_transform, is_test=True)

        val_loader = DataLoader(val_dataset, 1, num_workers = 2, shuffle=False)

    return train_loader, val_loader
                        
from ray.tune.integration.wandb import WandbLoggerCallback

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger


def train_net(config):
    lr = config['lr']
    momentum = config['momentum']
    weight_decay = config['weight_decay']
    t_max = config['t_max']
    base_lr = config['base_lr']

    num_classes = 11 ###########

    # select the network architecture and config     
    
    create_net = create_mobilenetv1_ssd
    config_ = mobilenetv1_ssd_config
        
    # create data transforms for train/test/val

    train_loader, val_loader = get_data_loaders()

    # create the network

    net = create_net(num_classes)
    last_epoch = -1

    # freeze certain layers (if requested)
    base_net_lr = base_lr ####
    extra_layers_lr = lr ####
    
    params = [
        {'params': net.base_net.parameters(), 'lr': base_net_lr},
        {'params': itertools.chain(
            net.source_layer_add_ons.parameters(),
            net.extras.parameters()
        ), 'lr': extra_layers_lr},
        {'params': itertools.chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters()
        )}
    ]

    net.init_from_pretrained_ssd("/mnt/c/Users/14135/Desktop/pytorch-ssd/models/mobilenet-v1-ssd-mp-0_675.pth")

    # move the model to GPU
    net.to(DEVICE)

    # define loss function and optimizer
    criterion = MultiboxLoss(config_.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)

    optimizer = torch.optim.SGD(params, lr = lr, momentum = momentum, weight_decay = weight_decay)

    scheduler = CosineAnnealingLR(optimizer, t_max, last_epoch=last_epoch)

    epoch = 0
    while True:
        scheduler.step()
        train(train_loader, net, criterion, optimizer, device=DEVICE, epoch=epoch)
        
        val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)

        tune.report(val_loss = val_loss, val_reg_loss = val_regression_loss, val_cls_loss = val_classification_loss)

        epoch += 1
        
def hyp_search():
    sched = AsyncHyperBandScheduler()

    analysis = tune.run(
        train_net,
        metric = "val_loss",
        mode = "min",
        name = "exp",
        scheduler = sched,
        stop = {
            "training_iteration": 1
        },
        #resources_per_trial={"cpu": 2, "gpu": 1},  # set this for GPUs
        resources_per_trial = {'cpu': 1},
        num_samples = 1,
        config = {
            'lr': tune.loguniform(1e-4, 1e-1),
            'base_lr': tune.loguniform(1e-6, 1e-2),
            'momentum': tune.uniform(0.1, 0.9),
            'weight_decay': tune.loguniform(1e-6, 1e-3),
            't_max': tune.uniform(50, 200),
        },
        callbacks = [WandbLoggerCallback(
            project = "SDP_hypsearch_0",
            api_key_file = "wandbkey",
            log_config = True)]
    )

    print("Best config is:", analysis.best_config)

if __name__ == "__main__":
    hyp_search()