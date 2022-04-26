train_url=${0%%.sh}
train_url=${train_url:21}
python main_moco.py \
--train_url=../${train_url} \
--data_dir=/home/xuhaohang/imagenet/ \
--moco_dim=128 \
--moco_k=65536 \
--moco_m=0.999 \
--moco_t=0.12 \
--mlp=true \
--aug_plus=true \
--decay_method=cos \
--init_lr=0.03 \
--wd=4e-5 \
--batch_size=256 \
--num_workers=8 \
--end_epoch=200 \
--dist=true \
--nodes_num=1 \
--node_rank=0 \
--subgroup=8 \
--master_addr=127.0.0.1 \
--lam=0.5,0.5 \
--report_freq=100 \
--corr_npy=imgs_corr_swav_epoch400.npy \
--pretrain_alg=swav \
--pretrain_path=/home/xuhaohang/haohang/swav_RN50w2_400ep_pretrain.pth.tar \
--t_arch=resnet50w2 \
--s_arch=efficientnet-b0 \
# --resume=true \