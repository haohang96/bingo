python main_lincls.py \
--train_url=../SR18-TR50W2 \
--data_dir=/home/xuhaohang/toy_imagenet \
--arch=resnet18 \
--usupv_lr=0.015 \
--usupv_batch=4 \
--pretrained_epoch=0 \
--init_lr=30. \
--batch_size=256 \
--wd=0. \
--selected_feat_id=17 \
--decay_method=cos \
--end_epoch=120 \
#--resume=true \
