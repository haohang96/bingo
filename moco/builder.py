# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn

from absl import flags
from absl import app

FLAGS = flags.FLAGS

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, teacher_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.dim = dim

        # create the encoders
        # num_classes is the output fc dimension
        if 'eff' in FLAGS.s_arch:
            self.encoder_q = base_encoder(FLAGS.s_arch, num_classes=dim)
        else:
            self.encoder_q = base_encoder(num_classes=dim)

        self.encoder_k = teacher_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            dim_mlp_k = self.encoder_k.fc.weight.shape[1]

            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, 2048), nn.ReLU(), nn.Linear(2048, 128))

            # encoder_q < encoder_k
            if FLAGS.pretrain_alg == 'moco':
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp_k, 2048), nn.ReLU(), nn.Linear(2048, 128))

            if FLAGS.pretrain_alg == 'swav':
                self.encoder_k.fc = nn.Sequential(
                        nn.Linear(4096,8192),
                        nn.BatchNorm1d(8192),
                        nn.ReLU(),
                        nn.Linear(8192,128))


        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _group_batch_shuffle_ddp(self, x, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather, select_subg = group_concat_all_gather(x, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this # here means num_gpus per subg

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        # src rank in select_subg
        src_rank = (gpu_rank//nrank_per_subg)*nrank_per_subg + node_rank*ngpu_per_node
        torch.distributed.broadcast(idx_shuffle, src=src_rank, group=select_subg)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = gpu_rank%nrank_per_subg # gpu_rank in each select_subg
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _group_batch_unshuffle_ddp(self, x, idx_unshuffle, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather,_ = group_concat_all_gather(x, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = gpu_rank%nrank_per_subg # gpu_rank in each select_subg
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, add_posq, add_posk, mix_q, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            gpu_rank: gpu rank in one node(0 to FLAGS.ngpu)
            node_rank: node rank(0 to FLAGS.nodes_num)
            groups: a list of subgroup, enable shuffle bn in each sub-group
        Output:
            logits, targets
        """

        _bs = im_q.size(0)
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)


        add_posq = self.encoder_q(add_posq)
        add_posq = nn.functional.normalize(add_posq, dim=1)

        all_q = torch.cat([q.unsqueeze(1), add_posq.reshape(_bs, 1, self.dim)], dim=1)

        mix_q = self.encoder_q(mix_q)
        mix_q = nn.functional.normalize(mix_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._group_batch_shuffle_ddp(im_k, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._group_batch_unshuffle_ddp(k, idx_unshuffle, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)


            add_posk, add_pos_idx_unshuffle = self._group_batch_shuffle_ddp(add_posk, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)
            add_posk = self.encoder_k(add_posk)
            add_posk = nn.functional.normalize(add_posk, dim=1)
            add_posk = self._group_batch_unshuffle_ddp(add_posk, add_pos_idx_unshuffle, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)



            all_k = torch.cat([k.unsqueeze(1), add_posk.reshape(_bs, 1, self.dim)], dim=1)

        # compute logits
        # Einstein sum is more intuitive
        logits = []
        for i in range(2): # loop a & pos_q
            q_i = all_q[:,i]
            
            _logits = self.logits_q_nk(q_i, all_k, self.queue)
            logits.append(_logits)


        mix_logits = self.logits_q_nk(mix_q, all_k, self.queue)


        # labels: positive key indicators
        labels = torch.zeros(logits[0].shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, mix_logits, labels


    def logits_q_nk(self, q, all_k, neg):
        _pos = torch.einsum('nmc,nc->nm', [all_k, q]).unsqueeze(-1) # N*2*1
        _neg = torch.einsum('nc,ck->nk',  [q, neg.clone().detach()]) # N*65536
        _neg = _neg.unsqueeze(1).repeat(1, 2, 1) # N*2*65536
        _logits = torch.cat([_pos, _neg], dim=2) # N*2*(65536+1)
        _logits /= self.T
        return _logits

# utils
@torch.no_grad()
def group_concat_all_gather(tensor, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    assert ngpu_per_node//nrank_per_subg == len(groups)
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(nrank_per_subg)]
    select_subg_idx = gpu_rank // nrank_per_subg
    select_subg = groups[select_subg_idx]
    torch.distributed.all_gather(tensors_gather, tensor, select_subg, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output, select_subg

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

