'''
Created on Mar 8, 2017

@author: Yikang
'''
import theano.tensor as tensor
import lasagne
from lasagne import init, nonlinearities


class Upsampling(lasagne.layers.Layer):
    
    def __init__(self, incoming, ratio, num_input_channels=None, **kwargs):
        super(Upsampling, self).__init__(incoming, **kwargs)
        self.ratio = ratio
        self.num_input_channels = num_input_channels

    def get_output_for(self, input, **kwargs):
        return tensor.nnet.abstract_conv.bilinear_upsampling(input, self.ratio, 
                                                             num_input_channels=self.num_input_channels)

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] *= self.ratio
        output_shape[-2] *= self.ratio
        return tuple(output_shape)
    
class ApplyMask(lasagne.layers.Layer):
    
    def __init__(self, incoming, mask, **kwargs):
        super(ApplyMask, self).__init__(incoming, **kwargs)
        self.mask = mask

    def get_output_for(self, input, **kwargs):
        return input * self.mask[None, None, :, :]
    
class ApplyNoise(lasagne.layers.Layer):
    
    def __init__(self, incoming, mask, **kwargs):
        super(ApplyNoise, self).__init__(incoming, **kwargs)
        self.mask = mask
        self.rng = tensor.shared_randomstreams.RandomStreams(seed=132468)

    def get_output_for(self, input, **kwargs):
        return input * self.mask[None, None, :, :] + self.rng.normal(input.shape) * (1. - self.mask[None, None, :, :])
    
class ApplyCaption(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(ApplyCaption, self).__init__(incomings, **kwargs)
        
    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]
        
    def get_output_for(self, inputs, **kwargs):
        img = inputs[0]
        cap = inputs[1]
        
        return tensor.set_subtensor(img[:,:,3:5,3:5], cap)
    
class ChooseLayer(lasagne.layers.MergeLayer):
    
    def __init__(self, incomings, **kwargs):
        super(ChooseLayer, self).__init__(incomings, **kwargs)
        
    def get_output_shape_for(self, input_shapes, chs_idx=0):
        return input_shapes[chs_idx]

    def get_output_for(self, inputs, chs_idx=0, **kwargs):
        return inputs[chs_idx]