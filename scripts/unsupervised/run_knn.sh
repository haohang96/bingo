python compute_knn.py \
--arch=resnet50w2 \
--batch_size=256 \
--data_dir=/home/xuhaohang/toy_imagenet/ \
--ngpu=4 \
--clus_pos_num=5 \
--ckpt_name=/home/xuhaohang/haohang/swav_RN50w2_400ep_pretrain.pth.tar \
--corr_name=imgs_corr_swav_epoch400.npy

