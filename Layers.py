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