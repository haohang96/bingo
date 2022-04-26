#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import shutil
import time
import math
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from classifiers.mul_classifier import MulClassifier as Classifier
from arch.efficientnet_pytorch.model_cls import EfficientNet
from arch import resnet_cls
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_bool('evaluate', False, '')

# default params for ModelArts
flags.DEFINE_string('train_url', '../moco_v2', 'path to output files(ckpt and log) on S3 or normal filesystem')
flags.DEFINE_string('data_url', '', 'path to datasets only on S3, only need on ModelArts')
flags.DEFINE_string('init_method', '', 'accept default flags of modelarts, nothing to do')

# params for dataset path
flags.DEFINE_string('data_dir', '/cache/dataset', 'path to datasets on S3 or normal filesystem used in dataloader')

# params for unsupervised folder
flags.DEFINE_string('unsupervised_folder', '', '')
flags.DEFINE_float('usupv_lr', 0.03, '')
flags.DEFINE_integer('usupv_batch', 256, '')
flags.DEFINE_integer('pretrained_epoch', 200, '')

# params for optimizer #
flags.DEFINE_integer('seed', None, 'seed for initializing training.')
flags.DEFINE_float('init_lr', 30., '')
flags.DEFINE_float('momentum', 0.9, '')
flags.DEFINE_float('wd', 0., '')
flags.DEFINE_integer('batch_size', 256, '')
flags.DEFINE_integer('num_workers', 32, '')
flags.DEFINE_integer('end_epoch', 100, 'total epochs')
flags.DEFINE_list('schedule', [60, 80], 'epochs when lr need drop')
flags.DEFINE_float('lr_decay', 0.1, 'scale factor for lr drop')
flags.DEFINE_enum('decay_method', 'step', ['step', 'cos'], 'default step for moco lincls')
flags.DEFINE_float('min_lr', 0.01, 'need when decay method is cos')

# params for classifier arch #
flags.DEFINE_string('arch', 'resnet18', '')
flags.DEFINE_list('selected_feat_id', [14, 15, 16, 17], '')
flags.DEFINE_string('pool_type', 'avg', '')
flags.DEFINE_integer('nchannels', 512, '')

# params for resume #
flags.DEFINE_bool('resume', False, '') 
flags.DEFINE_integer('resume_epoch', None, '')

# params for hardware
flags.DEFINE_bool('dist', True, 'DistributedDataparallel or no-dist mode, no-dist mode is only for debug')
flags.DEFINE_integer('nodes_num', 1, 'machine num')
flags.DEFINE_integer('ngpu', 4, 'ngpu per node')
flags.DEFINE_integer('world_size', 4, 'FLAGS.nodes_num*FLAGS.ngpu')
flags.DEFINE_integer('node_rank', 0, 'rank of machine, 0 to nodes_num-1')
flags.DEFINE_integer('rank', 0, 'rank of total threads, 0 to FLAGS.world_size-1')
flags.DEFINE_string('master_addr', '127.0.0.1', 'addr for master node')
flags.DEFINE_string('master_port', '2345', 'port for master node')

# params for log and save #
flags.DEFINE_integer('report_freq', 100, '') 
flags.DEFINE_integer('save_freq', 10, '') 

best_acc1 = 0


def adjust_learning_rate_pro(optimizer, epoch, log):
    """Decay the learning rate based on pycontrast way, used with lr_warmup"""
    lr = FLAGS.init_lr
    final_lr = FLAGS.min_lr
    # need sub FLAGS.warmup when use cos sche
    period = FLAGS.end_epoch
    step = epoch

    lr = final_lr + (lr-final_lr) * (1. + math.cos(math.pi * step / period))/2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    log.logger.info('==> Setting model optimizer lr = %.6f'%(lr))

def main(argv):
    del argv
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    # Prepare Workspace Folder #
    FLAGS.unsupervised_folder = os.path.join(FLAGS.train_url, 'unsupervised', 'lr-%s_batch-%s'
        %(FLAGS.usupv_lr, FLAGS.usupv_batch))
    FLAGS.train_url = os.path.join(FLAGS.train_url, 'classification', 'lr-%s_batch-%s'
        %(FLAGS.init_lr, FLAGS.batch_size))
    ############################
    if FLAGS.dist:

        FLAGS.ngpu = torch.cuda.device_count()
        FLAGS.world_size = FLAGS.ngpu * FLAGS.nodes_num
        os.environ['MASTER_ADDR'] = FLAGS.master_addr
        os.environ['MASTER_PORT'] = FLAGS.master_port
        if os.path.exists('tmp.cfg'):
            os.remove('tmp.cfg')
        FLAGS.append_flags_into_file('tmp.cfg')
        mp.spawn(main_worker, nprocs=FLAGS.ngpu, args=())








def main_worker(gpu_rank):
    global best_acc1
    # Prepare FLAGS #
    FLAGS._parse_args(FLAGS.read_flags_from_files(['--flagfile=./tmp.cfg']), True)
    FLAGS.mark_as_parsed()
    FLAGS.rank = FLAGS.node_rank * FLAGS.ngpu + gpu_rank # rank among FLAGS.world_size
    FLAGS.batch_size = FLAGS.batch_size // FLAGS.world_size
    FLAGS.num_workers = FLAGS.num_workers // FLAGS.ngpu
    # filter string list in flags to target format(int)
    tmp = FLAGS.schedule
    if isinstance(tmp[0], str):
        for i in range(len(tmp)):
            tmp[i] = int(tmp[i])
    FLAGS.schedule = tmp
    tmp = FLAGS.selected_feat_id
    if isinstance(tmp[0], str):
        for i in range(len(tmp)):
            tmp[i] = int(tmp[i])
    FLAGS.selected_feat_id = tmp

    from utils import Log, AverageMeter, ProgressMeter, accuracy, save_ckpt, adjust_learning_rate
    ############################
    # Set Log File #
    log = Log(FLAGS.train_url)

    ############################
    # Initial Log content #
    log.logger.info('Selected feat for lincls: %s'%(FLAGS.selected_feat_id))
    log.logger.info('Initialize optimizer: {\'decay_method: %s, batch_size(per GPU):%-4d, init_lr: %-.3f, momentum: %-.3f, weight_decay: %-.5f, lr_sche: %s, total_epoch: %-3d, num_workers(per GPU): %d, world_size: %d, rank:%d\'}'
        %(FLAGS.decay_method, FLAGS.batch_size, FLAGS.init_lr, FLAGS.momentum, \
        FLAGS.wd, FLAGS.schedule, FLAGS.end_epoch, \
        FLAGS.num_workers, FLAGS.world_size, FLAGS.rank))
    ############################


    # suppress printing if not master
    if gpu_rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass


    # Create DataLoader #
    traindir = os.path.join(FLAGS.data_dir, 'train')
    valdir = os.path.join(FLAGS.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=FLAGS.world_size, rank=FLAGS.rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=FLAGS.batch_size, shuffle=(train_sampler is None),
        num_workers=FLAGS.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=FLAGS.batch_size, shuffle=False,
        num_workers=FLAGS.num_workers, pin_memory=True)
    nbatch_per_epoch = len(train_loader)
    ############################


    # Create Model #  
    from classifiers.cls_opt import net_opt_cls
    log.logger.info('Selected feat info: %s'%(net_opt_cls))
    if 'eff' in FLAGS.arch:
        model = EfficientNet.from_name(FLAGS.arch, num_classes=1000)
    else:
        model = resnet_cls.__dict__[FLAGS.arch]()
    net_opt_cls[-1]['nchannels'] = FLAGS.nchannels
    net = Classifier(net_opt_cls)
    # log.logger.info(model)
    # log.logger.info(net)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=FLAGS.world_size,
        rank=FLAGS.rank)
    torch.cuda.set_device(gpu_rank)
    model.cuda()
    net.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu_rank])
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[gpu_rank])
    ############################
    # Create Optimizer #
    criterion = nn.CrossEntropyLoss().cuda(gpu_rank) 
    optimizer = torch.optim.SGD(net.parameters(), lr=FLAGS.init_lr, 
                                momentum=FLAGS.momentum, 
                                weight_decay=FLAGS.wd)
    ############################
    # Load Unsupervised Pretrained ckpt #
    pretrained_ckpt_path = os.path.join(FLAGS.unsupervised_folder,
        'ckpt_%d.pth.tar'%(FLAGS.pretrained_epoch))
    pretrained_ckpt = torch.load(pretrained_ckpt_path, map_location=torch.device('cpu'))
    log.logger.info("Load unsupervised pretrained ckpt '{}'".format(pretrained_ckpt_path))
    log.logger.info('Load unsupervised pretrained ckpt from %3d'%(pretrained_ckpt['epoch']-1))
    # rename moco pre-trained keys
    pretrained_state_dict = pretrained_ckpt['state_dict']
    for k in list(pretrained_state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            # pretrained_state_dict[k[len("module.encoder_q."):]] = pretrained_state_dict[k]
            pretrained_state_dict[k.replace('encoder_q.', '')] = pretrained_state_dict[k]
        # delete renamed or unused k
        del pretrained_state_dict[k]
    msg = model.load_state_dict(pretrained_state_dict, strict=False)
    print('Missing Keys when load unsupervised pretrained model', 
        msg.missing_keys)
    assert set(msg.missing_keys) == {"module.fc.weight", "module.fc.bias"}
    ############################
    # Resume Checkpoints #
    start_epoch = 0
    if FLAGS.resume:
        ckpt_path = os.path.join(FLAGS.train_url, 'ckpt.pth.tar')
        if FLAGS.resume_epoch is not None:
            ckpt_path = os.path.join(FLAGS.train_url, 'ckpt_%s.pth.tar'%(FLAGS.resume_epoch))
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log.logger.info('==> Load ckpt from epoch %3d'%(start_epoch-1))
    cudnn.benchmark = True


    if FLAGS.evaluate:
        assert FLAGS.resume is True
        validate(val_loader, model, net, criterion)
        return

    for epoch in range(start_epoch, FLAGS.end_epoch):
        train_sampler.set_epoch(epoch)
        if FLAGS.decay_method == 'step':
            adjust_learning_rate(optimizer, epoch, log)
        if FLAGS.decay_method == 'cos':
            adjust_learning_rate_pro(optimizer, epoch, log)

        # train for one epoch
        train(train_loader, model, net, criterion, optimizer, epoch, gpu_rank, log)

        # evaluate on validation set
        acc1 = validate(val_loader, model, net, criterion, gpu_rank, log)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_ckpt({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            }, epoch, FLAGS.save_freq, is_best)
        if epoch == start_epoch:
            sanity_check(model.state_dict(), pretrained_ckpt_path)


def train(train_loader, model, net, criterion, optimizer, epoch, gpu_rank, log):
    from utils import Log, AverageMeter, ProgressMeter, accuracy, save_ckpt, adjust_learning_rate
    model.eval()
    net.train()
    losses = AverageMeter('Loss', ':.4e')
    acc = []
    for i in range(len(FLAGS.selected_feat_id)):
        acc.append(AverageMeter('Acc@1', ':6.2f'))
    nbatch_per_epoch = len(train_loader)


    for batch_idx, (images, target) in enumerate(train_loader):

        images = images.cuda(gpu_rank, non_blocking=True)
        target = target.cuda(gpu_rank, non_blocking=True)

        # compute output
        with torch.no_grad():
            outputs = model(images)
        outputs = net(outputs)
        loss_total = None
        prec = []
        for i in range(len(outputs)):
            loss_this = criterion(outputs[i], target)
            loss_total = loss_this if (loss_total is None) else (loss_total + loss_this)
            prec.append(accuracy(outputs[i].data, target.data))

        for i in range(len(outputs)):
            acc[i].update(prec[i][0].item(), images.size(0))

        losses.update(loss_total.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if batch_idx % FLAGS.report_freq == 0:
            log.logger.info('==> Iter[%3d][%4d/%4d] loss : %2.5f Acc : %s'%
                (epoch, batch_idx, nbatch_per_epoch, loss_total, [prec[i][0].item() for i in range(len(outputs))]))

    log.logger.info('==> Training stats: Iter[%3d] loss : %2.5f Acc : %s'%
                (epoch, losses.avg, [acc[i].avg for i in range(len(outputs))]))




def validate(val_loader, model, net, criterion, gpu_rank, log):
    from utils import Log, AverageMeter, ProgressMeter, accuracy, save_ckpt, adjust_learning_rate
    model.eval()
    net.eval()
    losses = AverageMeter('Loss', ':.4e')
    acc = []
    for i in range(len(FLAGS.selected_feat_id)):
        acc.append(AverageMeter('Acc@1', ':6.2f'))



    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(val_loader):
            images = images.cuda(gpu_rank, non_blocking=True)
            target = target.cuda(gpu_rank, non_blocking=True)

            # compute output
            outputs = model(images)
            outputs = net(outputs)

            loss_total = None
            prec = []
            for i in range(len(outputs)):
                loss_this = criterion(outputs[i], target)
                loss_total = loss_this if (loss_total is None) else (loss_total+loss_this)
                prec.append(accuracy(outputs[i].data, target.data))

            for i in range(len(outputs)):
                acc[i].update(prec[i][0].item(), images.size(0))

            losses.update(loss_total.item(), images.size(0))

    log.logger.info('== Evaluating stats : loss = %3.5f Acc = %s'
        %(losses.avg, [acc[i].avg for i in range(len(outputs))]))


    np_acc = np.array([acc[i].avg for i in range(len(outputs))])

    # return max acc among all classifiers
    return np.max(np_acc)


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")




if __name__ == '__main__':
    app.run(main)
