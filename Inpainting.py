'''
Created on Feb 28, 2017

@author: Yikang
'''

import time

import numpy
import theano
import theano.tensor as tensor
import lasagne
import PIL.Image as Image

from Utils import *

from DataLoader import Mscoco
from PixelConvLayer import pixelConvLayer
from Layers import ApplyMask, ChooseLayer, ApplyNoise

rng = numpy.random.RandomState(seed=123456)

class Inpainting(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.history_errs = []
        self.best_p = None
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
        
    def Encode(self, x, mask):
        print 'Initialling encoder...'
        
        input_layer = lasagne.layers.InputLayer((None, 64, 64, 3), x)
        layer = lasagne.layers.dimshuffle(input_layer, [0,3,1,2])
        layer = lasagne.layers.BatchNormLayer(layer, beta=None, gamma=None)
        
        def encoder_layer(input, mask, num_units, ksize=(3,3), pooling=True):
            conv = lasagne.layers.Conv2DLayer(input, num_units, ksize, pad='same')
            conv = lasagne.layers.BatchNormLayer(conv, beta=None, gamma=None)
            if pooling:
                conv = lasagne.layers.Conv2DLayer(conv, num_units, (2,2), stride=2)
                conv = lasagne.layers.BatchNormLayer(conv, beta=None, gamma=None)
                mask = tensor.signal.pool.pool_2d(mask, (2,2), ignore_border=True)
            output = ApplyMask(conv, mask=mask)
            print output.output_shape
            
            return output, mask
        
        layer, mask = encoder_layer(layer, mask, num_units=64, ksize=(3,3), pooling=True) #32*32
        layer, mask = encoder_layer(layer, mask, num_units=128) #16*16
        layer, mask = encoder_layer(layer, mask, num_units=256) #8*8
#         layer, mask = encoder_layer(layer, mask, num_units=512) #4*4
        layer = ApplyNoise(layer, mask)
        
        return layer
    
    def Decode(self, encoder):
        print 'Initialling decoder...'
        
        layer = lasagne.layers.DenseLayer(encoder, 64, num_leading_axes=2)
        layer = lasagne.layers.reshape(layer, ([0], [1], 8, 8))
        layer = lasagne.layers.BatchNormLayer(layer, beta=None, gamma=None)
        print layer.output_shape
        
        def decode_layer(input, input_units, output_units, 
                         nonlinearity=lasagne.nonlinearities.rectify, 
                         ksize=(3,3), depool=True):
            if depool:
                input = lasagne.layers.Deconv2DLayer(input, input_units, (2, 2), stride=2)
                input = lasagne.layers.BatchNormLayer(input, beta=None, gamma=None)
                
            deconv = lasagne.layers.Deconv2DLayer(input, output_units, ksize, crop='same', 
                                                  nonlinearity=nonlinearity)
            if output_units != 3:
                deconv = lasagne.layers.BatchNormLayer(deconv, beta=None, gamma=None)
            
            print deconv.output_shape
            return deconv
        
#         layer0 = decode_layer(reshape1, 1024, 512)
#         layer = decode_layer(layer, 512, 256)
        layer = decode_layer(layer, 256, 128)
        layer = decode_layer(layer, 128, 64)
        layer = decode_layer(layer, 64, 3, nonlinearity=lasagne.nonlinearities.tanh, depool=False)
        
        draft = lasagne.layers.dimshuffle(layer, [0,2,3,1])
        output = draft
        print output.output_shape
        
        return output
    
    def Discriminate(self, input):
        print 'Initialling discriminator...'
        
        input_layer = lasagne.layers.InputLayer((None, 64, 64, 3), input)
        layer = lasagne.layers.dimshuffle(input_layer, [0,3,1,2])
        
        def Discriminate_layer(input, num_units, ksize=(3,3), pooling=True):
            conv = lasagne.layers.Conv2DLayer(input, num_units, ksize, pad='same', nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2))
            conv = lasagne.layers.BatchNormLayer(conv, beta=None, gamma=None)
            if pooling:
                conv = lasagne.layers.Conv2DLayer(conv, num_units, (2,2), stride=2, nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2))
                conv = lasagne.layers.BatchNormLayer(conv, beta=None, gamma=None)
            print conv.output_shape
            
            return conv
        
        layer = Discriminate_layer(layer, num_units=32) #32*32
        layer = Discriminate_layer(layer, num_units=64) #16*16
        layer = Discriminate_layer(layer, num_units=128) #8*8
        layer = Discriminate_layer(layer, num_units=256) #4*4
#         layer5 = Discriminate_layer(layer4, num_units=256) #2*2
        
#         layer = lasagne.layers.DenseLayer(layer, 1024, num_leading_axes=1, 
#                                           nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2))
#         layer = lasagne.layers.BatchNormLayer(layer, beta=None, gamma=None)
#         print layer.output_shape
        output = lasagne.layers.DenseLayer(layer, 1, num_leading_axes=1,
                                           nonlinearity=lasagne.nonlinearities.sigmoid)
        
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
        
    def init_model(self):
        print 'Initialing model'
        self.x = tensor.tensor4(name='x', dtype='float32')
        self.y = tensor.tensor4(name='y', dtype='float32')
        self.mask = tensor.matrix(name='mask', dtype='float32')
        
        self.encoder = self.Encode(self.x, self.mask)
        
        self.decoder = self.Decode(self.encoder)
        
        self.y_hat = lasagne.layers.get_output(self.decoder)
        self.cost = self.Cost_l2(self.y_hat, self.y[:, 16:48, 16:48])
        self.f_cost = theano.function([self.x, self.y, self.mask], self.cost, name='Reconstruction cost function')
        
        self.generate = tensor.set_subtensor(self.x[:, 16:48, 16:48], self.y_hat)
        self.original = tensor.set_subtensor(self.x[:, 16:48, 16:48], self.y[:, 16:48, 16:48])
        
        self.disciminator = self.Discriminate(self.y)
        print self.disciminator.output_shape
        real_score = lasagne.layers.get_output(self.disciminator)
        fake_score = lasagne.layers.get_output(self.disciminator, self.generate)
        
#         self.cost_Dsc = (real_soore - fake_score).mean()
        self.cost_Dsc = (lasagne.objectives.binary_crossentropy(real_soore, 1) + lasagne.objectives.binary_crossentropy(fake_score, 0)).mean()
        self.f_cost_Dsc = theano.function([self.x, self.y, self.mask], self.cost_Dsc, name='Discriminative cost function')
        
#         self.cost_Gen = fake_score.mean()
        self.cost_Gen = lasagne.objectives.binary_crossentropy(fake_score, 1).mean()
        self.f_cost_Gen = theano.function([self.x, self.mask], self.cost_Gen, name='Generative cost function')
        
        self.test_cost = self.Cost_l2(self.y_hat, self.y[:, 16:48, 16:48])
        self.f_test_cost = theano.function([self.x, self.y, self.mask], self.test_cost, name='cost function')
        
        self.generator_params = lasagne.layers.get_all_params(self.decoder, trainable=True)
        print len(self.generator_params)
        self.discriminator_params = lasagne.layers.get_all_params(self.disciminator, trainable=True)
        print len(self.discriminator_params)
        
        self.f_yhat = theano.function([self.x, self.mask], self.y_hat, name='yhat')
        self.f_generate = theano.function([self.x, self.mask], self.generate, name='Generate')
        self.f_original = theano.function([self.x, self.y], self.original, name='Original')
        print 'Done initial'
        
    def show_examples(self, dataset):
        y_hat = self.f_yhat(numpy.array(dataset.valid_x[:10], dtype='float32') / 255. * 2. - 1., dataset.mask)
        generate = self.f_generate(numpy.array(dataset.valid_x[:10], dtype='float32') / 255. * 2. - 1., dataset.mask)
        original = self.f_original(numpy.array(dataset.valid_x[:10], dtype='float32') / 255. * 2. - 1., 
                                   numpy.array(dataset.valid_y[:10], dtype='float32') / 255. * 2. - 1.)
        
        y_hat = y_hat.reshape([y_hat.shape[0] * y_hat.shape[1], y_hat.shape[2], y_hat.shape[3]])
        generate = generate.reshape([generate.shape[0] * generate.shape[1], generate.shape[2], generate.shape[3]])
        original = original.reshape([original.shape[0] * original.shape[1], original.shape[2], original.shape[3]])
        
        fig = numpy.int64((numpy.concatenate([original, generate], axis=1) + 1.) / 2. * 255.) 
        fig = numpy.clip(fig, 0, 255).astype('uint8')
        Image.fromarray(fig, mode='RGB').show()
        
    def learn_model(self, dataset, 
                    batch_size=32, 
                    valid_batch_size=128, 
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
        self.f_update = theano.function([self.x, self.y, self.mask], self.cost, updates=updates)
            
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
    
                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, y = dataset.prepare_data(x, y)
                    n_samples += x.shape[0]
    
                    cost = self.f_update(x, y, dataset.mask)
    
                    if numpy.isnan(cost) or numpy.isinf(cost):
                        print 'bad cost detected: ', cost
                        return 1., 1., 1.
    
                    if numpy.mod(uidx, dispFreq) == 0:
                        nowtime = time.time()
                        print 'Epoch ', eidx, 'Update ', uidx - eidx * batchnum, '/', batchnum, 'Cost ', cost, \
                              'Time cost ', nowtime - start_time, 'Expected epoch time cost ', (nowtime - start_time) * batchnum / uidx
    
                    if numpy.mod(uidx, validFreq) == 0:
                        #train_err = pred_error(f_decode, prepare_data, train, kf)
                        valid_decode_err = error(self.f_test_cost, dataset.prepare_data, dataset.valid_x, dataset.valid_y, dataset.mask, kf_valid)
    
                        self.history_errs.append(valid_decode_err)
    
                        if (self.best_p is None 
                            or 
                            valid_decode_err <= numpy.array(self.history_errs).min()):
    
                            del self.best_p
                            self.best_p = lasagne.layers.get_all_param_values(self.decoder)
                            bad_counter = 0
                            
                            self.save_model(saveto)
                            
                            self.show_examples(dataset)
    
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
        
    def learn_model_GAN(self, dataset, 
                    batch_size=64, 
                    valid_batch_size=128, 
                    saveto='params/model', 
                    optimizer=lasagne.updates.adam,
                    patience=5, 
                    lrate=0.0002, 
                    dispFreq=50, 
                    validFreq=-1, 
                    saveFreq=-1, 
                    max_epochs=40,
                    n_critic=1,
                    clip_params=0.01,
                    ):
        print 'Computing gradient...'
        updates = optimizer(self.cost, self.generator_params, lrate)
        self.f_update = theano.function([self.x, self.y, self.mask], self.cost, updates=updates)

        updates_Dsc = optimizer(self.cost_Dsc, self.discriminator_params, lrate, beta1=0.5)
#         for param in lasagne.layers.get_all_params(self.disciminator, trainable=True, regularizable=True):
#             updates_Dsc[param] = tensor.clip(updates_Dsc[param], -clip_params, clip_params)
        self.f_update_Dsc = theano.function([self.x, self.y, self.mask], self.cost_Dsc, updates=updates_Dsc)
        
        updates_Gen = optimizer(self.cost_Gen, self.generator_params, lrate, beta1=0.5)
        self.f_update_Gen = theano.function([self.x, self.mask], self.cost_Gen, updates=updates_Gen)
            
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
                    
                    for i in range(n_critic):
                        random_idx = rng.randint(len(kf))
                        _, train_index_sample = kf[random_idx]
                        x = [dataset.train_x[t]for t in train_index_sample]
                        
                        random_idx = rng.randint(len(kf))
                        _, train_index_sample = kf[random_idx]
                        y = [dataset.train_y[t]for t in train_index_sample]
                        
                        if len(x) != batch_size or len(y) != batch_size:
                            print 'unmatched sample'
                            continue
        
                        # Get the data in numpy.ndarray format
                        # This swap the axis!
                        # Return something of shape (minibatch maxlen, n samples)
                        x, y = dataset.prepare_data(x, y)
        
                        cost_Dsc = self.f_update_Dsc(x, y, dataset.mask)
                        
                        if numpy.isnan(cost_Dsc) or numpy.isinf(cost_Dsc):
                            print 'bad cost detected: ', cost_Dsc
                            return 1., 1., 1.
    
                    # Select the random examples for this minibatch
                    x = [dataset.train_x[t] #+ dataset.mask * numpy.random.randint(0,256,size=[64,64,3])
                         for t in train_index]
                    y = [dataset.train_y[t]for t in train_index]
     
                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, y = dataset.prepare_data(x, y)
                    n_samples += x.shape[0]
                     
                    cost_Gen = self.f_update_Gen(x, dataset.mask)
    
                    if numpy.isnan(cost_Gen) or numpy.isinf(cost_Gen):
                        print 'bad cost detected: ', cost_Gen
                        return 1., 1., 1.

                    cost = self.f_update(x, y ,dataset.mask)
#                     cost = -1
    
                    if numpy.mod(uidx, dispFreq) == 0:
                        nowtime = time.time()
                        print 'Epoch ', eidx, 'Update ', uidx - eidx * batchnum, '/', batchnum, 'Cost Gen', cost_Gen, 'Cost Dsc', cost_Dsc, 'Cost L2', cost, \
                              'Time cost ', nowtime - start_time, 'Expected epoch time cost ', (nowtime - start_time) * batchnum / uidx
    
                    if numpy.mod(uidx, validFreq) == 0:
                        #train_err = pred_error(f_decode, prepare_data, train, kf)
                        valid_decode_err = error(self.f_cost_Dsc, dataset.prepare_data, dataset.valid_x, dataset.valid_y, dataset.mask, kf_valid)
    
                        self.history_errs.append(valid_decode_err)
    
                        if (self.best_p is None 
                            or 
                            valid_decode_err <= numpy.array(self.history_errs).min()):
    
                            del self.best_p
                            self.best_p = lasagne.layers.get_all_param_values(self.decoder)
                            bad_counter = 0
                            
                            self.save_model(saveto)
                            
                        self.show_examples(dataset)
    
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
        self.show_examples()
        return valid_decode_err
        
dataset = Mscoco('D:/workspace/Data/inpainting/')
model = Inpainting()
# model.learn_model(dataset, saveto='params/model.npz')
# model.load_model(saveto='params/model.npz')
# model.show_examples(dataset)
model.learn_model_GAN(dataset, saveto='params/model_GAN.npz')
