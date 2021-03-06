'''
Created on Feb 28, 2017

@author: Yikang
'''

import time

import numpy
import theano
import theano.tensor as tensor
import lasagne
import lasagne.layers.dnn
import PIL.Image as Image

from Utils import *

from DataLoader import Mscoco
from PixelConvLayer import pixelConvLayer
from Layers import ApplyMask, ChooseLayer, ApplyNoise, ApplyCaption

rng = numpy.random.RandomState(seed=123456)

class Inpainting(object):
    '''
    classdocs
    '''


    def __init__(self, nwords):
        '''
        Constructor
        '''
        self.history_errs = []
        self.best_p = None
        self.nwords = nwords
        self.init_model()
        
    def save_model(self, saveto):
        print('Saving...')
        numpy.savez(saveto, 
                    history_errs=self.history_errs, 
                    params=self.best_p)
        print('Done')
        
    def load_model(self, saveto):
        print('Loading...')
#         print('initial model...')
#         self.init_model()
#         print('load parameters')
        pp = numpy.load(saveto)
        lasagne.layers.set_all_param_values(self.decoder, pp['params'])
        print('Done')
        
    def Cap_Encode(self, c, cmask):
        input_layer = lasagne.layers.InputLayer((None, None), c)
        mask_layer = lasagne.layers.InputLayer((None, None), cmask)
        emb_layer = lasagne.layers.EmbeddingLayer(input_layer, self.nwords, 64)
        lstm_layer = lasagne.layers.LSTMLayer(emb_layer, 256, mask_input=mask_layer, 
                                              peepholes=False, only_return_final=True)
        output_layer = lasagne.layers.DenseLayer(lstm_layer, 2*2*256)
        output_layer = lasagne.layers.ReshapeLayer(output_layer, [[0], 256, 2, 2])
        
        return output_layer
        
        
    def Encode(self, x, mask, c_layer, add_noise=True):
        print 'Initialling encoder...'
        
        input_layer = lasagne.layers.InputLayer((None, 64, 64, 3), x)
        layer = lasagne.layers.dimshuffle(input_layer, [0,3,1,2])
        layer = lasagne.layers.dnn.batch_norm_dnn(layer, beta=None, gamma=None)
        
        def encoder_layer(input, mask, num_units, ksize=(3,3), pooling=True):
            conv = lasagne.layers.Conv2DLayer(input, num_units, ksize, pad='same')
            conv = lasagne.layers.dnn.batch_norm_dnn(conv, beta=None, gamma=None)
            mask = tensor.signal.pool.pool_2d(mask, (3,3), ignore_border=True, stride=(1,1), pad=(1,1))
            if pooling:
                conv = lasagne.layers.Conv2DLayer(conv, num_units, (2,2), stride=2)
                conv = lasagne.layers.dnn.batch_norm_dnn(conv, beta=None, gamma=None)
                mask = tensor.signal.pool.pool_2d(mask, (2,2), ignore_border=True)
            output = ApplyMask(conv, mask=mask)
            print output.output_shape
            
            return output, mask
        
        layer, mask = encoder_layer(layer, mask, num_units=64) #32*32
        layer, mask = encoder_layer(layer, mask, num_units=128) #16*16
        layer, mask = encoder_layer(layer, mask, num_units=256) #8*8
#         layer, mask = encoder_layer(layer, mask, num_units=512) #4*4
#         if add_noise:
#             layer = ApplyNoise(layer, mask)
        layer = ApplyCaption([layer, c_layer])
        
        return layer
    
    def Decode(self, encoder):
        print 'Initialling decoder...'
        
        layer = lasagne.layers.DenseLayer(encoder, 64, num_leading_axes=2)
        layer = lasagne.layers.reshape(layer, ([0], [1], 8, 8))
        layer = lasagne.layers.dnn.batch_norm_dnn(layer, beta=None, gamma=None)
        print layer.output_shape
        
        def decode_layer(input, input_units, output_units, 
                         nonlinearity=lasagne.nonlinearities.rectify, 
                         ksize=(3,3), depool=True):
            if depool:
                input = lasagne.layers.Deconv2DLayer(input, input_units, (2, 2), stride=2)
                input = lasagne.layers.dnn.batch_norm_dnn(input, beta=None, gamma=None)
                
            deconv = lasagne.layers.Deconv2DLayer(input, output_units, ksize, crop='same', 
                                                  nonlinearity=nonlinearity)
            if output_units != 3:
                deconv = lasagne.layers.dnn.batch_norm_dnn(deconv, beta=None, gamma=None)
            
            print deconv.output_shape
            return deconv
        
#         layer = decode_layer(layer, 512, 256)
        layer = decode_layer(layer, 256, 128)
        layer = decode_layer(layer, 128, 64)
        layer = decode_layer(layer, 64, 3, nonlinearity=lasagne.nonlinearities.tanh, depool=False)
        
        draft = lasagne.layers.dimshuffle(layer, [0,2,3,1])
        output = draft
        print output.output_shape
        
        return output
    
    def Discriminate(self, input, output_nonlin=lasagne.nonlinearities.sigmoid):
        print 'Initialling discriminator...'

        if isinstance(input, lasagne.layers.Layer):
            layer = input
        else:
            input_layer = lasagne.layers.InputLayer((None, 64, 64, 3), input)
            layer = lasagne.layers.dimshuffle(input_layer, [0,3,1,2])
        
        def Discriminate_layer(input, num_units, ksize=(3,3), pooling=True):
            conv = lasagne.layers.Conv2DLayer(input, num_units, ksize, pad='same', nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2))
            conv = lasagne.layers.dnn.batch_norm_dnn(conv, beta=None, gamma=None)
            if pooling:
                conv = lasagne.layers.Conv2DLayer(conv, num_units, (2,2), stride=2, nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2))
                conv = lasagne.layers.dnn.batch_norm_dnn(conv, beta=None, gamma=None)
            print conv.output_shape
            
            return conv
        
        layer = Discriminate_layer(layer, num_units=64) #32*32
        layer = Discriminate_layer(layer, num_units=128) #16*16
        layer = Discriminate_layer(layer, num_units=256) #8*8
        layer = Discriminate_layer(layer, num_units=512) #4*4
#         layer5 = Discriminate_layer(layer4, num_units=256) #2*2
        
#         layer = lasagne.layers.DenseLayer(layer, 1024, num_leading_axes=1, 
#                                           nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2))
#         layer = lasagne.layers.dnn.batch_norm_dnn(layer, beta=None, gamma=None)
#         print layer.output_shape
        output = lasagne.layers.DenseLayer(layer, 1, num_leading_axes=1,
                                           nonlinearity=output_nonlin)
        
        return output
    
    def Cost_loglikelihood(self, py, y):
        py = py.reshape([py.shape[0]*py.shape[1]*py.shape[2]*py.shape[3], 256])
        y = y.flatten()
        offset = 1e-20
        cost = -tensor.log(py[y] + offset).mean()
        return cost
    
    def Cost_l2(self, y_hat, y):
        cost = tensor.mean(tensor.sqr(y_hat - y))
        return cost
    
    def show_examples(self, dataset, batchsize, figname='example'):
        x, y, c, cmask = dataset.prepare_data([dataset.valid_x[t] for t in range(batchsize)], 
                                        [dataset.valid_y[t] for t in range(batchsize)], 
                                        [dataset.valid_c[t] for t in range(batchsize)])
        
        generate = self.f_generate(x, dataset.mask, c, cmask)
        original = self.f_original(x, y)
     
        fig = numpy.int64((numpy.concatenate([original, generate], axis=1) + 1.) / 2. * 255.) 
        fig = fig.reshape([batchsize/8, 8, fig.shape[1], fig.shape[2], fig.shape[3]])
        figs = [numpy.concatenate([subfig for subfig in fig[i]], axis=1) for i in range(batchsize/8)]
        fig = numpy.concatenate(figs, axis=0)
#         fig = numpy.concatenate([subfig for subfig in fig], axis=2)
        fig = numpy.clip(fig, 0, 255).astype('uint8')
        Image.fromarray(fig, mode='RGB').show()
        Image.fromarray(fig, mode='RGB').save(figname, format='png')
        
    def init_model(self):
        print 'Initialing model'
        self.x = tensor.tensor4(name='x', dtype='float32')
        self.y = tensor.tensor4(name='y', dtype='float32')        
        self.mask = tensor.matrix(name='mask', dtype='float32')
        
        self.c = tensor.matrix(name='c', dtype='int64')
        self.cmask = tensor.matrix(name='cmask', dtype='float32')
        
        self.c_encoder = self.Cap_Encode(self.c, self.cmask)
        
        self.encoder = self.Encode(self.x, self.mask, self.c_encoder, add_noise=False)
        
        self.decoder = self.Decode(self.encoder)
        
        self.y_hat = lasagne.layers.get_output(self.decoder, 
                                               batch_norm_update_averages=True)
        self.cost = self.Cost_l2(self.y_hat, self.y[:, 16:48, 16:48])
        
        self.generate = tensor.set_subtensor(self.x[:, 16:48, 16:48], self.y_hat)
        self.original = tensor.set_subtensor(self.x[:, 16:48, 16:48], self.y[:, 16:48, 16:48])
        
        self.test_cost = self.Cost_l2(self.y_hat, self.y[:, 16:48, 16:48])
        self.f_test_cost = theano.function([self.x, self.y, self.mask, self.c, self.cmask], self.test_cost, name='cost function')
        
        self.generator_params = lasagne.layers.get_all_params(self.decoder, trainable=True)
        print len(self.generator_params)
        
        self.y_hat_output = lasagne.layers.get_output(self.decoder, batch_norm_update_averages=True)
        self.generate_output = tensor.set_subtensor(self.x[:, 16:48, 16:48], self.y_hat_output)
        self.f_yhat = theano.function([self.x, self.mask, self.c, self.cmask], self.y_hat_output, name='yhat')
        self.f_generate = theano.function([self.x, self.mask, self.c, self.cmask], self.generate_output, name='Generate')
        self.f_original = theano.function([self.x, self.y], self.original, name='Original')
        print 'Done initial'
        
    def learn_model(self, dataset, 
                    batch_size=32, 
                    valid_batch_size=64, 
                    saveto='params/model', 
                    optimizer=lasagne.updates.adam,
                    patience=5, 
                    lrate=0.0003, 
                    dispFreq=100, 
                    validFreq=-1, 
                    saveFreq=-1, 
                    max_epochs=10,
                    ):
        print 'Computing gradient...'
        updates = optimizer(self.cost, self.generator_params, lrate)
        self.f_update = theano.function([self.x, self.y, self.mask, self.c, self.cmask], self.cost, updates=updates)
            
        print('Optimization')
        kf_valid = get_minibatches_idx(len(dataset.valid_x), valid_batch_size)
    
        print("%d train examples" % len(dataset.train_x))
        print("%d valid examples" % len(dataset.valid_x))
    
        bad_counter = 0
    
        batchnum = len(dataset.train_x) // batch_size
        if validFreq == -1:
            validFreq = len(dataset.train_x) // batch_size
        if saveFreq == -1:
            saveFreq = len(dataset.train_x) // batch_size
            
        uidx = 0  # the number of update done
        estop = False  # early stop
        start_time = time.time()
        try:
            for eidx in range(max_epochs):
                n_samples = 0
    
                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(len(dataset.train_x), batch_size, shuffle=True)
    
                epoch_start_time = time.time()
                for _, train_index in kf:
                    uidx += 1
    
                    # Select the random examples for this minibatch
                    x = [dataset.train_x[t] #+ dataset.mask * numpy.random.randint(0,256,size=[64,64,3])
                         for t in train_index]
                    y = [dataset.train_y[t]for t in train_index]
                    c = [dataset.train_c[t]for t in train_index]
    
                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, y, c, cmask = dataset.prepare_data(x, y, c)
                    n_samples += x.shape[0]
    
                    cost = self.f_update(x, y, dataset.mask, c, cmask)
    
                    if numpy.isnan(cost) or numpy.isinf(cost):
                        print 'bad cost detected: ', cost
                        return 1., 1., 1.
    
                    if numpy.mod(uidx, dispFreq) == 0:
                        nowtime = time.time()
                        print 'Epoch ', eidx, 'Update ', uidx - eidx * batchnum, '/', batchnum, 'Cost ', cost, \
                              'Time cost ', nowtime - start_time, 'Expected epoch time cost ', (nowtime - start_time) * batchnum / uidx
    
                    if numpy.mod(uidx, validFreq) == 0:
                        #train_err = pred_error(f_decode, prepare_data, train, kf)
                        valid_decode_err = error(self.f_test_cost, dataset.prepare_data, 
                                                 dataset.valid_x, dataset.valid_y, dataset.valid_c,
                                                 dataset.mask, kf_valid)
    
                        self.history_errs.append(valid_decode_err)
    
                        if (self.best_p is None 
                            or 
                            valid_decode_err <= numpy.array(self.history_errs).min()):
    
                            del self.best_p
                            self.best_p = lasagne.layers.get_all_param_values(self.decoder)
                            bad_counter = 0
                            
                            self.save_model(saveto)
                            
                            self.show_examples(dataset, batch_size, 'figures/AE_epoch' + str(eidx) + '.png')
    
                        print 'Valid decode error:', valid_decode_err
    
                        if (len(self.history_errs) > patience and
                            valid_decode_err >= numpy.array(self.history_errs)[:-patience].min()):
                            bad_counter += 1
                            if bad_counter > patience:
                                print('Early Stop!')
                                estop = True
                                break
    
                print 'Seen %d samples' % n_samples
    
                if estop:
                    break
    
        except KeyboardInterrupt:
            print "Training interupted"
    
        end_time = time.time()
        if self.best_p is not None:
            lasagne.layers.set_all_param_values(self.decoder, self.best_p)
        else:
            self.best_p = lasagne.layers.get_all_param_values(self.decoder)
    
        #kf_train_sorted = get_minibatches_idx(len(train), batch_size)
        #train_err = pred_error(f_sentiment, prepare_data, train, kf_train_sorted)
        valid_decode_err = error(self.f_test_cost, dataset.prepare_data, dataset.valid_x, dataset.valid_y, dataset.mask, kf_valid)
    
        print 'Valid decode error:', valid_decode_err
        if saveto:
            self.save_model(saveto)
        print 'The code run for %d epochs, with %f sec/epochs' % (
            (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
        print 'Training took %.1fs' % (end_time - start_time)
        self.show_examples(dataset)
        return valid_decode_err
        
if __name__ == '__main__':
    dataset = Mscoco('../Data/inpainting/')
    model = Inpainting(max(dataset.wdict.values()) + 1)
    # model.learn_model(dataset, saveto='params/model.npz')
    # model.load_model(saveto='params/model.npz')
    # model.show_examples(dataset)
    model.learn_model(dataset, saveto='params/model_AE.npz')
