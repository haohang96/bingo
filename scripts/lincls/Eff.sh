python main_lincls.py \
--train_url=../SEffB0-TR50W2 \
--data_dir=/home/xuhaohang/toy_imagenet \
--usupv_lr=0.015 \
--usupv_batch=4 \
--pretrained_epoch=0 \
--init_lr=3. \
--min_lr=0.001 \
--batch_size=256 \
--arch=efficientnet-b0 \
--nchannels=1280 \
--selected_feat_id=17 \
--wd=0. \
--decay_method=cos \
--end_epoch=120 \
#--resume=true \
