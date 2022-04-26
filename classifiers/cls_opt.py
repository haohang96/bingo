from absl import flags
from absl import app

FLAGS = flags.FLAGS
pool_type = FLAGS.pool_type


net_opt_cls = [None] * 18
net_opt_cls[1]  = {'pool_type':pool_type, 'nchannels':64,   'in_feat_size':56, 'out_feat_size':14, 'num_class':1000}
net_opt_cls[4]  = {'pool_type':pool_type, 'nchannels':256,  'in_feat_size':56, 'out_feat_size':14, 'num_class':1000}
net_opt_cls[8]  = {'pool_type':pool_type, 'nchannels':512,  'in_feat_size':28, 'out_feat_size':7,  'num_class':1000}
net_opt_cls[12] = {'pool_type':pool_type, 'nchannels':1024, 'in_feat_size':14, 'out_feat_size':7,  'num_class':1000}
net_opt_cls[13] = {'pool_type':pool_type, 'nchannels':1024, 'in_feat_size':14, 'out_feat_size':7,  'num_class':1000}
net_opt_cls[14] = {'pool_type':pool_type, 'nchannels':1024, 'in_feat_size':14, 'out_feat_size':7,  'num_class':1000}
net_opt_cls[15] = {'pool_type':pool_type, 'nchannels':2048, 'in_feat_size':7,  'out_feat_size':7,  'num_class':1000}
net_opt_cls[16] = {'pool_type':pool_type, 'nchannels':2048, 'in_feat_size':7,  'out_feat_size':7,  'num_class':1000}
net_opt_cls[17] = {'pool_type':pool_type, 'nchannels':2048, 'in_feat_size':7,  'out_feat_size':1,  'num_class':1000}

len_feat = len(FLAGS.selected_feat_id)
net_opt_cls = [net_opt_cls[FLAGS.selected_feat_id[i]] for i in range(len_feat)]
