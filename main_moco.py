#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import pdb
import random
import shutil
import time
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

import moco.loader
import moco.builder
import arch

from arch.efficientnet_pytorch.model import EfficientNet
from moco import folder
from absl import flags
from absl import app

FLAGS = flags.FLAGS


# default params for ModelArts
flags.DEFINE_string('train_url', '../moco_v2', 'path to output files(ckpt and log)')
flags.DEFINE_string('data_url', '', 'path to datasets only on S3, only need on ModelArts')

# params for dataset path
flags.DEFINE_string('data_dir', '/cache/dataset', 'path to datasets on S3 or normal filesystem used in dataloader')
flags.DEFINE_integer('dataset_len', 1281167, '')

# params for workspace folder
flags.DEFINE_string('cache_ckpt_folder', '', 'folder path to ckpt files in /cache, only need on ModelArts')

# params for specific moco config #
flags.DEFINE_integer('moco_dim', 128, 'feature dim for constrastive loss')
flags.DEFINE_integer('moco_k', 65536, 'queue size; number of negative keys (default: 65536)')
flags.DEFINE_float('moco_m', 0.999, 'moco momentum of updating key encoder (default: 0.999)')
flags.DEFINE_float('moco_t', 0.2, 'softmax temperature (moco_v1 default: 0.07)')

# params for moco v2 #
flags.DEFINE_bool('mlp', True, 'if projection head is used, set True for v2')
flags.DEFINE_bool('aug_plus', True, 'set True for v2')
flags.DEFINE_enum('decay_method', 'cos', ['step', 'cos'], 'set cos for v2')

# params for resume #
flags.DEFINE_bool('resume', False, '') 
flags.DEFINE_integer('resume_epoch', None, '') 

# params for optimizer #
flags.DEFINE_integer('seed', None, 'seed for initializing training.')
flags.DEFINE_float('init_lr', 0.03, '')
flags.DEFINE_float('momentum', 0.9, '')
flags.DEFINE_float('wd', 1e-4, '')
flags.DEFINE_integer('batch_size', 256, '')
flags.DEFINE_integer('num_workers', 32, '')
flags.DEFINE_integer('end_epoch', 200, 'total epochs')
flags.DEFINE_list('schedule', [120, 160], 'epochs when lr need drop')
flags.DEFINE_float('lr_decay', 0.1, 'scale factor for lr drop')

# params for hardware
flags.DEFINE_bool('dist', True, 'DistributedDataparallel or no-dist mode, no-dist mode is only for debug')
flags.DEFINE_integer('nodes_num', 1, 'machine num')
flags.DEFINE_integer('ngpu', 8, 'ngpu per node')
flags.DEFINE_integer('world_size', 8, 'FLAGS.nodes_num*FLAGS.ngpu')
flags.DEFINE_integer('node_rank', 0, 'rank of machine, 0 to nodes_num-1')
flags.DEFINE_integer('rank', 0, 'rank of total threads, 0 to FLAGS.world_size-1')
flags.DEFINE_string('master_addr', '127.0.0.1', 'addr for master node')
flags.DEFINE_string('master_port', '1234', 'port for master node')

# params for log and save
flags.DEFINE_integer('report_freq', 100, '')
flags.DEFINE_integer('save_freq', 10, '')

# params for group shuffle bn
flags.DEFINE_integer('subgroup', 8, 'num of ranks each subgroup contain, only subgroup=ngpu is tested (subgroup<ngpu has not beed tested, not recommened)' )

# params for cluster
flags.DEFINE_list('lam', [0.5, 0.5], 'trade-off coeff of multi pos sample')
flags.DEFINE_integer('clus_pos_num', 5, 'number of pos select by clustering, no include self')

# params for distillation
flags.DEFINE_string('s_arch', 'resnet18', 'student model name')
flags.DEFINE_string('t_arch', 'resnet50', 'student model name')
flags.DEFINE_string('corr_npy', 'imgs_corr_epoch800_MoCo.npy', '')
flags.DEFINE_string('pretrain_path', 'moco_v2_800ep_pretrain.pth.tar', 'name of pretrained ckpt')
flags.DEFINE_string('pretrain_alg', 'moco', 'name of pretrain algorithm')



class MulposCrossEntropy(nn.Module):
    def __init__(self):
        super(MulposCrossEntropy, self).__init__()
        self.ce = nn.CrossEntropyLoss().cuda()

    def forward(self, inputs, targets, lam):
        """
        Args:
            inputs: bs*2*(1+65536)
            targets: bs
        """
        num_loss = inputs.size(1)
        losses = []
        for i in range(num_loss):
            losses.append(self.ce(inputs[:,i], targets))

        total_loss = 0
        for i in range(num_loss):
            total_loss += lam[i] * losses[i]

        return total_loss

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
    FLAGS.train_url = os.path.join(FLAGS.train_url, 'unsupervised', 'lr-%s_batch-%s'
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

    else: # single-gpu mode for debug
        model = moco.builder.MoCo(
            resnet50,
            FLAGS.moco_dim, FLAGS.moco_k, FLAGS.moco_m, FLAGS.moco_t, FLAGS.mlp)



def main_worker(gpu_rank):
    # Prepare FLAGS #
    FLAGS._parse_args(FLAGS.read_flags_from_files(['--flagfile=./tmp.cfg']), True)
    FLAGS.mark_as_parsed()
    FLAGS.rank = FLAGS.node_rank * FLAGS.ngpu + gpu_rank # rank among FLAGS.world_size
    FLAGS.batch_size = FLAGS.batch_size // FLAGS.world_size
    FLAGS.num_workers = FLAGS.num_workers // FLAGS.ngpu
    assert FLAGS.subgroup == FLAGS.ngpu # before FLAGS.subgroup < FLAGS.ngpu is tested, not use such setting
    # filter string list in flags to target format(int)
    tmp = FLAGS.schedule
    if isinstance(tmp[0], str):
        for i in range(len(tmp)):
            tmp[i] = int(tmp[i])
    FLAGS.schedule = tmp
    tmp = FLAGS.lam
    if isinstance(tmp[0], str):
        for i in range(len(tmp)):
            tmp[i] = float(tmp[i])
    FLAGS.lam = tmp

    from utils import Log, AverageMeter, ProgressMeter, accuracy, save_ckpt, adjust_learning_rate, cutmix
    ############################
    # Set Log File #
    log = Log(FLAGS.train_url)

    ############################
    # Initial Log content #
    log.logger.info('Moco specific configs: {\'moco_dim: %-5d, moco_k: %-5d, moco_m: %-.5f, moco_t: %-.5f\'}'
        %(FLAGS.moco_dim, FLAGS.moco_k, FLAGS.moco_m, FLAGS.moco_t))
    log.logger.info('Projection head: %s (True means mocov2, False means mocov1)'
        %(FLAGS.mlp))
    log.logger.info('Initialize optimizer: {\'decay_method: %s, batch_size(per GPU): %-4d, init_lr: %-.3f, momentum: %-.3f, weight_decay: %-.5f, lr_sche: %s, total_epoch: %-3d, num_workers(per GPU): %d, world_size: %d, rank: %d\'}'
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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if FLAGS.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]

    train_dataset = folder.ImageFolder(
        traindir,
        transforms.Compose(augmentation))
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=FLAGS.world_size, rank=FLAGS.rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=FLAGS.batch_size, shuffle=(train_sampler is None),
        num_workers=FLAGS.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    nbatch_per_epoch = len(train_loader)
    FLAGS.dataset_len = len(train_dataset)

    ############################
    # Create Model #
    if 'eff' in FLAGS.s_arch: # if use efficientnet as student
        student_model = EfficientNet.from_name
    else:
        student_model = arch.__dict__[FLAGS.s_arch]
    teacher_model = arch.__dict__[FLAGS.t_arch]
    model = moco.builder.MoCo(
        student_model,
        teacher_model,
        FLAGS.moco_dim, FLAGS.moco_k, 
        FLAGS.moco_m, FLAGS.moco_t, 
        FLAGS.mlp)
    # log.logger.info(model)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=FLAGS.world_size,
        rank=FLAGS.rank)
    torch.cuda.set_device(gpu_rank)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu_rank])
    groups = []
    # for example, FLAGS.nodes_num=2, FLAGS.ngpu=4, FLAGS.subgroup=4
    # groups = [[0,1,2,3]] in node_rank = 0
    # groups = [[4,5,6,7]] in node_rank = 1
    for i in range(FLAGS.nodes_num):
        for j in range(FLAGS.ngpu//FLAGS.subgroup):
            ranks = []
            for k in range(FLAGS.subgroup):
                ranks.append(j*FLAGS.subgroup + k + i*FLAGS.ngpu)
                _group = dist.new_group(ranks=ranks) 
            if FLAGS.node_rank == i:
                print('ranks: ', ranks)
                groups.append(_group)
    ############################
    # Create Optimizer #
    criterion = MulposCrossEntropy().cuda(gpu_rank)
    optimizer = torch.optim.SGD(model.parameters(), FLAGS.init_lr,
                                momentum=FLAGS.momentum,
                                weight_decay=FLAGS.wd)
    ############################
    # Resume Checkpoints #
    start_epoch = 0
    if FLAGS.resume:
        ckpt_path = os.path.join(FLAGS.train_url, 'ckpt.pth.tar')
        if FLAGS.resume_epoch is not None:
            ckpt_path = os.path.join(FLAGS.train_url, 'ckpt_%s.pth.tar'\
                %(FLAGS.resume_epoch))

        loc = 'cuda:{}'.format(gpu_rank)
        checkpoint = torch.load(ckpt_path, map_location=loc)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log.logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpt_path, checkpoint['epoch']-1))
    cudnn.benchmark = True
    ############################
    # Resume Pretrained Encoder K #
    pretrained_ckpt = torch.load(FLAGS.pretrain_path, map_location=torch.device('cpu'))

    # 1. MoCo style
    if FLAGS.pretrain_alg == 'moco':
        pretrained_state_dict = pretrained_ckpt['state_dict']
        for k in list(pretrained_state_dict.keys()):
            if k.startswith('module.encoder_q'):
                pretrained_state_dict[k.replace('encoder_q', 'encoder_k')] = pretrained_state_dict[k]
            del pretrained_state_dict[k]
        msg = model.load_state_dict(pretrained_state_dict, strict=False)
        print('Mising Keys when load unsupervsied pretrained model: ', msg.missing_keys)


    # 2. Swav stype
    if FLAGS.pretrain_alg == 'swav':
        pretrained_state_dict = pretrained_ckpt
        for k in list(pretrained_state_dict.keys()):
            new_k = k[:7] + 'encoder_k.' + k[7:]
            if 'projection_head' in new_k:
                new_k = new_k.replace('projection_head', 'fc')
                print(new_k)
            pretrained_state_dict[new_k] = pretrained_state_dict[k]
            del pretrained_state_dict[k]
        msg = model.load_state_dict(pretrained_state_dict, strict=False)
        print('Mising Keys when load unsupervsied pretrained model: ', msg.missing_keys)


    ############################
    # Start Train Process #
    imgs_corr = np.load('./imgs_corr/%s'%(FLAGS.corr_npy))
    imgs_corr = imgs_corr[:, :FLAGS.clus_pos_num]
    dist.barrier()
    train_dataset.set_imgs_corr(imgs_corr)

    optimizer.zero_grad()
    for epoch in range(start_epoch, FLAGS.end_epoch):

        log.logger.info('Training epoch [%3d/%3d]'%(epoch, FLAGS.end_epoch))
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, log)
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        for i, (query, key, add_posq, add_posk) in enumerate(train_loader):
            _bs = key.size(0)
            query = query.cuda(gpu_rank, non_blocking=True)
            key = key.cuda(gpu_rank, non_blocking=True)
            add_posq = add_posq.cuda(gpu_rank, non_blocking=True)
            add_posk = add_posk.cuda(gpu_rank, non_blocking=True)

            mix_query, mix_lam = cutmix(query.clone(), add_posq.clone())
            mix_lam = [mix_lam, 1 - mix_lam]

            # compute output
            output, mix_out, target = model(
                im_q=query, 
                im_k=key, 
                add_posq=add_posq,
                add_posk=add_posk,
                mix_q = mix_query,
                gpu_rank=gpu_rank, 
                node_rank=FLAGS.node_rank, 
                ngpu_per_node=FLAGS.ngpu,
                nrank_per_subg=FLAGS.subgroup,
                groups=groups)

            loss = torch.tensor(0.).cuda()
            for out_idx in range(len(output)):
                soft_lam = np.zeros(2)
                for j in range(soft_lam.shape[0]):
                    if j == out_idx:
                        soft_lam[j] = FLAGS.lam[0] if epoch > 10 else 1
                    else:
                        soft_lam[j] = FLAGS.lam[1] if epoch > 10 else 0
            
                _loss = criterion(output[out_idx], target, soft_lam)
                loss += _loss

            mix_loss = criterion(mix_out, target, mix_lam)

            loss += mix_loss

            loss /= 3

            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output[0][:,0], target, topk=(1, 5))
            losses.update(loss.item(), _bs)
            top1.update(acc1[0], _bs)
            top5.update(acc5[0], _bs)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if i % FLAGS.report_freq == 0:
                progress.display(i, log)


        log.logger.info('==> Training stats: Iter[%3d] loss=%2.5f; top1: %2.3f; top5: %2.3f'%
            (epoch, losses.avg, top1.avg, top5.avg))

        save_ckpt({'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch+1,}, epoch, FLAGS.save_freq)
    #####################################        



if __name__ == '__main__':
    app.run(main)
