import theano
import theano.tensor as T
import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.cuda.dnn import dnn_conv
from generic_utils import *
import lasagne
from lasagne import nonlinearities

def floatX(num):
    if theano.config.floatX == 'float32':
        return numpy.float32(num)
    else:
        raise Exception("{} type not supported".format(theano.config.floatX))

srng = RandomStreams(seed=3732)


def uniform(stdev, size):
    """uniform distribution with the given stdev and size"""
    return numpy.random.uniform(
        low=-stdev * numpy.sqrt(3),
        high=stdev * numpy.sqrt(3),
        size=size
    ).astype(theano.config.floatX)

def bias_weights(length, initialization='zeros', param_list = None, name = ""):
    "theano shared variable for bias unit, given length and initialization"
    if initialization == 'zeros':
        bias_initialization = numpy.zeros(length).astype(theano.config.floatX)
    else:
        raise Exception("Not Implemented Error: {} initialization not implemented".format(initialization))

    bias =  theano.shared(
            bias_initialization,
            name=name
            )
    if param_list is not None:
        param_list.append(bias)

    return bias

'''
get_conv_2d_filter: Takes a filter_shape (a tuple/array of length 4) and returns corresponding convolution filter
masktype is optional.
param_list is mandatory. It appends all the parameters from the function to this list
'''
def get_conv_2d_filter(filter_shape, param_list = None, masktype = None, name = ""):
    fan_in = numpy.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
    w_std = numpy.sqrt(2.0 / (fan_in + fan_out))

    filter_init = uniform(w_std, filter_shape)


    if masktype is not None:
        filter_init *= floatX(numpy.sqrt(2.))

    conv_filter = theano.shared(filter_init, name = name)
    param_list.append(conv_filter)

    if masktype is not None:
        mask = numpy.ones(
            filter_shape,
            dtype=theano.config.floatX
            )

        for i in range(filter_shape[2]):
            for j in range(filter_shape[3]):
                if i > filter_shape[2]//2:
                    mask[:,:,i,j] = floatX(0.0)

                if i == filter_shape[2]//2 and j > filter_shape[3]//2:
                    mask[:,:,i,j] = floatX(0.0)

        if masktype == 'a':
            mask[:,:,filter_shape[2]//2,filter_shape[3]//2] = floatX(0.0)

        conv_filter = conv_filter*mask

    return conv_filter

class Layer:
    '''
    Generic Layer Template which all layers should inherit.
    Every layer should have a name and params attribute containing all
    trainable parameters for that layer.
    '''
    def __init__(self, name = ""):
        self.name = name
        self.params = []

    def get_params(self):
        return self.params


class Conv2D(Layer):
    """
    Basic convolution layer
    input_shape: (batch_size, input_channels, height, width)
    filter_size: int or (row, column)
    """
    def __init__(self, input_layer, input_channels, output_channels, filter_size, subsample = (1,1), border_mode='half', masktype = None, activation = None, name = ""):
        self.X = input_layer.output()
        self.name = name
        self.subsample = subsample
        self.border_mode = border_mode

        self.params = []

        if isinstance(filter_size, tuple):
            self.filter_shape = (output_channels, input_channels, filter_size[0], filter_size[1])
        else:
            self.filter_shape = (output_channels, input_channels, filter_size, filter_size)

        self.filter = get_conv_2d_filter(self.filter_shape, param_list = self.params, masktype = masktype, name=name+'.filter')

        self.bias = bias_weights((output_channels,), param_list = self.params, name = name+'.b')

        self.activation = activation


        conv_out = T.nnet.conv2d(self.X, self.filter, border_mode = self.border_mode, filter_flip=False)
        self.Y = conv_out + self.bias[None,:,None,None]
        if self.activation is not None:
            self.Y = self.activation(self.Y)

    def output(self):
        return self.Y


class pixelConvLayer(lasagne.layers.Layer):
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
    def __init__(self, incoming, input_dim, DIM, Q_LEVELS = None, num_layers = 6, nonlinearity=nonlinearities.rectify, name='PixelCNN', **kwargs):

        super(pixelConvLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        ''' for first layer filter size should be 7 x 7 '''
        self.input_dim = input_dim
        self.DIM = DIM
        self.Q_LEVELS = Q_LEVELS
        self.num_layers = num_layers
        self.name = name
        
        
    def get_output_for(self, input, **kwargs):

        ''' 
        masked filter_size x filter_size convolution for vertical stack effect can be achieved 
        by just convolving the image with (filter_size // 2) + 1, filter_size) filter, 
        padding filter_size // 2 + 1 rows and filter_size // 2 0s columns on both sides of images with 0s.
        This is easy to see that in this case first row in the ouput does not depend on the image, 
        second row depends only on the first row of the image and so on. The final effect is anything in the i'th row
        of the output use information only upto i-1th row in the input.
        '''
        filter_size = 7 # for first layer
        
        input = input.dimshuffle(0,3,1,2)
        vertical_stack = Conv2D(
            WrapperLayer(input), 
            self.input_dim,
            self.DIM, 
            ((filter_size // 2) + 1, filter_size), 
            masktype=None, 
            border_mode=(filter_size // 2 + 1, filter_size // 2), 
            name= self.name + ".vstack1",
            activation = None
            )

        out_v = vertical_stack.output()

        '''
        while generating i'th row we can only use information upto i-1th row in the vertical stack.
        Horizontal stack gets input from vertical stack as well as previous layer.
        '''
        vertical_and_input_stack = T.concatenate([out_v[:,:,:-(filter_size//2)-2,:], input], axis=1)

        '''horizontal stack is straight forward. For first layer, I have used masked convolution as
             we are not allowed to see the pixel we would generate. 
        '''

        horizontal_stack = Conv2D(
            WrapperLayer(vertical_and_input_stack), 
            self.input_dim+self.DIM, self.DIM, 
            (1,filter_size), 
            border_mode = (0,filter_size//2), 
            masktype='a', 
            name = self.name + ".hstack1",
            activation = None
            )

        self.tparams = vertical_stack.params + horizontal_stack.params

        X_h = horizontal_stack.output() #horizontal stack output
        X_v = out_v[:,:,1:-(filter_size//2) - 1,:] #vertical stack output

        filter_size = 3 #all layers beyond first has effective filtersize 3

        '''
        one run of the loop implements four convolutions mentioned in figure 2 of the image
        with residual connection added on the horizontal stack.
        Two convolutions are just linear transformations of the faeture maps as convolution filter size is (1,1)
        '''
        for i in range(self.num_layers - 2):
            vertical_stack = Conv2D(
                WrapperLayer(X_v), 
                self.DIM, 
                self.DIM, 
                ((filter_size // 2) + 1, filter_size), 
                masktype = None, 
                border_mode = (filter_size // 2 + 1, filter_size // 2), 
                name= self.name + ".vstack{}".format(i+2),
                activation = None
                )
            v2h = Conv2D(
                vertical_stack, 
                self.DIM, 
                self.DIM, 
                (1,1), 
                masktype = None, 
                border_mode = 'valid', 
                name= self.name + ".v2h{}".format(i+2),
                activation = None
                )
            out_v = v2h.output()
            vertical_and_prev_stack = T.concatenate([out_v[:,:,:-(filter_size//2)-2,:], X_h], axis=1)

            horizontal_stack = Conv2D(
                WrapperLayer(vertical_and_prev_stack),
                self.DIM*2, 
                self.DIM,
                (1, (filter_size // 2) + 1), 
                border_mode = (0, filter_size // 2), 
                masktype = None, 
                name = self.name + ".hstack{}".format(i+2),
                activation = self.nonlinearity
                )

            h2h = Conv2D(
                horizontal_stack,
                self.DIM, 
                self.DIM,
                (1, 1), 
                border_mode = 'valid', 
                masktype = None, 
                name = self.name + ".h2hstack{}".format(i+2),
                activation = self.nonlinearity
                )

            self.tparams += (vertical_stack.params + horizontal_stack.params + v2h.params + h2h.params)

            X_v = self.nonlinearity(vertical_stack.output()[:,:,1:-(filter_size//2) - 1,:])
            X_h = h2h.output()[:,:,:,:-(filter_size//2)] + X_h #residual connection added

        '''single fully connected layer.'''

        combined_stack1 = Conv2D(
                WrapperLayer(X_h), 
                self.DIM, 
                self.DIM, 
                (1, 1), 
                masktype = None, 
                border_mode = 'valid', 
                name=self.name+".combined_stack1",
                activation = self.nonlinearity
                )

        if self.Q_LEVELS is None:
            out_dim = self.input_dim
        else:
            out_dim = self.input_dim*self.Q_LEVELS

        combined_stack2 = Conv2D(
                combined_stack1, 
                self.DIM, 
                out_dim, 
                (1, 1), 
                masktype = None, 
                border_mode = 'valid', 
                name=self.name+".combined_stack2",
                activation = None
                )

        self.tparams += (combined_stack1.params + combined_stack2.params)

        pre_final_out = combined_stack2.output().dimshuffle(0,2,3,1)

        if self.Q_LEVELS is None:
            self.Y = pre_final_out
        else:
            # pre_final_out = pre_final_out.dimshuffle(0,1,2,3,'x')
            old_shape = pre_final_out.shape
            self.Y = pre_final_out.reshape((old_shape[0], old_shape[1], old_shape[2],  old_shape[3]//self.Q_LEVELS, -1))
            
        for p in self.tparams:
            self.add_param(p, shape=p.get_value().shape)

        return self.Y
    
    def get_output_shape_for(self, input_shape):
        if self.Q_LEVELS is None:
            out_dim = self.input_dim
        else:
            out_dim = self.input_dim*self.Q_LEVELS
        return tuple(list(input_shape)[:-1] + [out_dim])
    

class WrapperLayer(Layer):
    def __init__(self, X, name=""):
        self.params = []
        self.name = name
        self.X = X

    def output(self):
        return self.X

class Softmax(Layer):
    def __init__(self, input_layer,  name=""):
        self.input_layer = input_layer
        self.name = name
        self.params = []
        self.X = self.input_layer.output()
        self.input_shape = self.X.shape

    def output(self):
        return T.nnet.softmax(self.X.reshape((-1,self.input_shape[self.X.ndim-1]))).reshape(self.input_shape)