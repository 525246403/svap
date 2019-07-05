from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import mxnet as mx
from mxnet import ndarray as nd

class MxnetExtractor:
    def __init__(self, model_path, batch_size, image_size, output_name, data_name='data', ctx=mx.gpu(0)):
        print('image_size', image_size)
        nets = []
        vec = model_path.split(',')
        prefix = vec[0]
        epoch = 0
        if len(vec) > 1:
            epoch = int(vec[1])
        print('loading',prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        #arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
        all_layers = sym.get_internals()
        sym = all_layers[output_name]
        model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
        model.bind(data_shapes=[(data_name, ( batch_size, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        # save context
        self.model = model
        self.batch_size = batch_size

    
    def extract(self, x_batch):
        x_batch = np.transpose(x_batch, [0,3,1,2])
        _label = nd.ones( (self.batch_size,) )
        nd_x = nd.array(x_batch)
        #print(nd_x)
        #db = mx.io.DataBatch(data=(nd_x, ), label=(_label, ))
        db = mx.io.DataBatch(data=(nd_x, ) )
        self.model.forward(db, is_train=False)
        net_out = self.model.get_outputs()
        feat = net_out[0].asnumpy()
        #feat = np.squeeze(feat)
        #print(feat.shape)
        return feat
        
        
    def close(self):
        pass
        


