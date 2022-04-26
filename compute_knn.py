import subprocess
import argparse
import builtins
import math
import os
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

import arch

from moco import cluster_folder
from clustering import compute_feat, knn
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string('arch', 'resnet50', 'network name')
flags.DEFINE_integer('batch_size', 32, 'batch size per gpu')

# params for dataset path
flags.DEFINE_string('data_dir', '/cache/dataset', 'path to datasets on S3 or normal filesystem used in dataloader')
flags.DEFINE_integer('dataset_len', 1281167, '')

# params for hardware
flags.DEFINE_integer('nodes_num', 1, 'machine num')
flags.DEFINE_integer('ngpu', 4, 'ngpu per node')
flags.DEFINE_integer('world_size', 4, 'FLAGS.nodes_num*FLAGS.ngpu')
flags.DEFINE_integer('node_rank', 0, 'rank of machine, 0 to nodes_num-1')
flags.DEFINE_integer('rank', 0, 'rank of total threads, 0 to FLAGS.world_size-1')
flags.DEFINE_string('master_addr', '127.0.0.1', 'addr for master node')
flags.DEFINE_string('master_port', '1234', 'port for master node')

# params for cluster
flags.DEFINE_integer('clus_pos_num', 5, 'number of pos select by clustering, no include self')
flags.DEFINE_string('ckpt_name', '', 'name of checkpoint file')
flags.DEFINE_string('corr_name', 'imgs_corr_moco_epoch800.npy', 'name of saved npy file')


def main(argv):
    del argv
    ############################

    FLAGS.ngpu = torch.cuda.device_count()
    FLAGS.world_size = FLAGS.ngpu * FLAGS.nodes_num
    os.environ['MASTER_ADDR'] = FLAGS.master_addr
    os.environ['MASTER_PORT'] = FLAGS.master_port
    if os.path.exists('tmp.cfg'):
        os.remove('tmp.cfg')
    FLAGS.append_flags_into_file('tmp.cfg')
    mp.spawn(main_worker, nprocs=FLAGS.ngpu, args=())




def main_worker(gpu_rank):
    # Prepare FLAGS #
    FLAGS._parse_args(FLAGS.read_flags_from_files(['--flagfile=./tmp.cfg']), True)
    FLAGS.mark_as_parsed()
    FLAGS.rank = FLAGS.node_rank * FLAGS.ngpu + gpu_rank # rank among FLAGS.world_size
    ############################
    if gpu_rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    # Create DataLoader #
    traindir = os.path.join(FLAGS.data_dir, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    cluster_augmentation = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]
    cluster_dataset = cluster_folder.ImageFolder(traindir, transforms.Compose(cluster_augmentation))
    cluster_train_sampler = torch.utils.data.distributed.DistributedSampler(
        cluster_dataset, num_replicas=FLAGS.world_size, shuffle=False, rank=FLAGS.rank)
    cluster_loader = torch.utils.data.DataLoader(
        cluster_dataset, batch_size=FLAGS.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, sampler=cluster_train_sampler, drop_last=False)
    FLAGS.dataset_len = len(cluster_dataset)

    ############################
    # Create Model #
    model = arch.__dict__[FLAGS.arch]()
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
    ############################
    # Resume Checkpoints #
    start_epoch = 0
    ckpt_path = FLAGS.ckpt_name
    loc = 'cuda:{}'.format(gpu_rank)
    checkpoint = torch.load(ckpt_path, map_location=loc)
    # Rename pre-trained model keys 
    # 1. MoCo style pretrained ckpt
    '''
    pretrained_state_dict = checkpoint['state_dict']
    for k in list(pretrained_state_dict.keys()):
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            new_k = k.replace('encoder_q.', '')
            pretrained_state_dict[new_k] = pretrained_state_dict[k]
        del pretrained_state_dict[k]
    '''
    # 2. Swav style pretrained ckpt
    pretrained_state_dict = checkpoint
    ###################################
    


    msg = model.load_state_dict(pretrained_state_dict, strict=False)
    print(msg)

    
    cudnn.benchmark = True
    ############################
    # Start KNN Process #
    feats = compute_feat(model, cluster_loader, gpu_rank)
    if FLAGS.rank == 0:
        clus_out = knn(feats)
        np.save('./imgs_corr/%s'%(FLAGS.corr_name), clus_out.imgs_corr)

    dist.barrier()


if __name__ == '__main__':
    app.run(main)
