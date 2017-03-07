'''
Created on Mar 7, 2017

@author: Yikang
'''
import theano
import theano.tensor as T
import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.cuda.dnn import dnn_conv
from generic_utils import *

srng = RandomStreams(seed=3732)

class pixelConv(Layer):
    """
    This layer implements pixelCNN module which is mentioned in https://arxiv.org/abs/1606.05328
    Main diferences: activation is not gated in this implementation.
    Masking is not used except for the first horizontal stack. Instead, appropriate filter size 
    with appropriate shifting of output feature maps used get the same effect as that of masking. This
    has been described in detail in the second paragraph of section 2.2 of the paper. There is no blind spots.
    There are four convolutions per as described in figure 2. Left output and input in this image corresponds
    to vertical feature map and right output/input corresponds to horizontal feature map. There is residual 
    connection added on the horizontal stack
    input_shape: (batch_size, height, width, input_dim)
    output_shape: (batch_size, height, width, input_dim)
                when Q_LEVELS is None
                else 
                (batch_size, height, width, input_dim, Q_LEVELS)
    """
    def __init__(self, input_layer, input_dim, DIM, Q_LEVELS = None, num_layers = 6, activation='relu', name=""):

        if activation is None:
            apply_act = lambda r: r
        elif activation == 'relu':
            apply_act = T.nnet.relu
        elif activation == 'tanh':
            apply_act = T.tanh
        else:
            raise Exception("{} activation not implemented!!".format(activation))


        self.X = input_layer.output().dimshuffle(0,3,1,2)

        ''' for first layer filter size should be 7 x 7 '''
        filter_size = 7 # for first layer

        ''' 
        masked filter_size x filter_size convolution for vertical stack effect can be achieved 
        by just convolving the image with (filter_size // 2) + 1, filter_size) filter, 
        padding filter_size // 2 + 1 rows and filter_size // 2 0s columns on both sides of images with 0s.
        This is easy to see that in this case first row in the ouput does not depend on the image, 
        second row depends only on the first row of the image and so on. The final effect is anything in the i'th row
        of the output use information only upto i-1th row in the input.
        '''
        vertical_stack = Conv2D(
            WrapperLayer(self.X), 
            input_dim,
            DIM, 
            ((filter_size // 2) + 1, filter_size), 
            masktype=None, 
            border_mode=(filter_size // 2 + 1, filter_size // 2), 
            name= name + ".vstack1",
            activation = None
            )

        out_v = vertical_stack.output()

        '''
        while generating i'th row we can only use information upto i-1th row in the vertical stack.
        Horizontal stack gets input from vertical stack as well as previous layer.
        '''
        vertical_and_input_stack = T.concatenate([out_v[:,:,:-(filter_size//2)-2,:], self.X], axis=1)

        '''horizontal stack is straight forward. For first layer, I have used masked convolution as
             we are not allowed to see the pixel we would generate. 
        '''

        horizontal_stack = Conv2D(
            WrapperLayer(vertical_and_input_stack), 
            input_dim+DIM, DIM, 
            (1,filter_size), 
            border_mode = (0,filter_size//2), 
            masktype='a', 
            name = name + ".hstack1",
            activation = None
            )

        self.params = vertical_stack.params + horizontal_stack.params

        X_h = horizontal_stack.output() #horizontal stack output
        X_v = out_v[:,:,1:-(filter_size//2) - 1,:] #vertical stack output

        filter_size = 3 #all layers beyond first has effective filtersize 3

        '''
        one run of the loop implements four convolutions mentioned in figure 2 of the image
        with residual connection added on the horizontal stack.
        Two convolutions are just linear transformations of the faeture maps as convolution filter size is (1,1)
        '''
        for i in range(num_layers - 2):
            vertical_stack = Conv2D(
                WrapperLayer(X_v), 
                DIM, 
                DIM, 
                ((filter_size // 2) + 1, filter_size), 
                masktype = None, 
                border_mode = (filter_size // 2 + 1, filter_size // 2), 
                name= name + ".vstack{}".format(i+2),
                activation = None
                )
            v2h = Conv2D(
                vertical_stack, 
                DIM, 
                DIM, 
                (1,1), 
                masktype = None, 
                border_mode = 'valid', 
                name= name + ".v2h{}".format(i+2),
                activation = None
                )
            out_v = v2h.output()
            vertical_and_prev_stack = T.concatenate([out_v[:,:,:-(filter_size//2)-2,:], X_h], axis=1)

            horizontal_stack = Conv2D(
                WrapperLayer(vertical_and_prev_stack),
                DIM*2, 
                DIM,
                (1, (filter_size // 2) + 1), 
                border_mode = (0, filter_size // 2), 
                masktype = None, 
                name = name + ".hstack{}".format(i+2),
                activation = activation
                )

            h2h = Conv2D(
                horizontal_stack,
                DIM, 
                DIM,
                (1, 1), 
                border_mode = 'valid', 
                masktype = None, 
                name = name + ".h2hstack{}".format(i+2),
                activation = activation
                )

            self.params += (vertical_stack.params + horizontal_stack.params + v2h.params + h2h.params)

            X_v = apply_act(vertical_stack.output()[:,:,1:-(filter_size//2) - 1,:])
            X_h = h2h.output()[:,:,:,:-(filter_size//2)] + X_h #residual connection added

        '''single fully connected layer.'''

        combined_stack1 = Conv2D(
                WrapperLayer(X_h), 
                DIM, 
                DIM, 
                (1, 1), 
                masktype = None, 
                border_mode = 'valid', 
                name=name+".combined_stack1",
                activation = activation
                )

        if Q_LEVELS is None:
            out_dim = input_dim
        else:
            out_dim = input_dim*Q_LEVELS

        combined_stack2 = Conv2D(
                combined_stack1, 
                DIM, 
                out_dim, 
                (1, 1), 
                masktype = None, 
                border_mode = 'valid', 
                name=name+".combined_stack2",
                activation = None
                )

        self.params += (combined_stack1.params + combined_stack2.params)

        pre_final_out = combined_stack2.output().dimshuffle(0,2,3,1)

        if Q_LEVELS is None:
            self.Y = pre_final_out
        else:
            # pre_final_out = pre_final_out.dimshuffle(0,1,2,3,'x')
            old_shape = pre_final_out.shape
            self.Y = pre_final_out.reshape((old_shape[0], old_shape[1], old_shape[2],  old_shape[3]//Q_LEVELS, -1))

    def output(self):
        return self.Y