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
from Layers import ApplyMask, ChooseLayer

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
        dimshuffle1 = lasagne.layers.dimshuffle(input_layer, [0,3,1,2])
        dimshuffle1 = lasagne.layers.BatchNormLayer(dimshuffle1)
        
        def encoder_layer(input, mask, num_units, ksize=(3,3)):
            conv = lasagne.layers.Conv2DLayer(input, num_units, ksize, pad='same')
            pool = lasagne.layers.Conv2DLayer(conv, num_units, (2,2), stride=2)
            input = lasagne.layers.BatchNormLayer(input)
            maskpool = tensor.signal.pool.pool_2d(mask, (2,2), ignore_border=True)
            output = ApplyMask(pool, mask=maskpool)
            print output.output_shape
            
            return output, maskpool
        
        layer1, mask1 = encoder_layer(dimshuffle1, mask, num_units=64, ksize=(3,3)) #32*32
        layer2, mask2 = encoder_layer(layer1, mask1, num_units=128) #16*16
        layer3, mask3 = encoder_layer(layer2, mask2, num_units=256) #8*8
        layer4, mask4 = encoder_layer(layer3, mask3, num_units=512) #4*4
#         layer5, mask5 = encoder_layer(layer4, mask4, num_units=1024) #2*2
        
#         conv5 = lasagne.layers.Conv2DLayer(pool4, 768, (3,3), pad='same')
#         pool5 = lasagne.layers.Pool2DLayer(conv5, (2,2))
#         print pool5.output_shape
        
        return layer4
    
    def Decode(self, encoder):
        print 'Initialling decoder...'
        
        channel_full1 = lasagne.layers.DenseLayer(encoder, 16, num_leading_axes=2)
        reshape1 = lasagne.layers.reshape(channel_full1, ([0], [1], 4, 4))
        print reshape1.output_shape
        
        def decode_layer(input, input_units, output_units, 
                         nonlinearity=lasagne.nonlinearities.rectify, ksize=(3,3), depool=True):
            input = lasagne.layers.BatchNormLayer(input)
            if depool:
                input = lasagne.layers.Deconv2DLayer(input, input_units, (2, 2), stride=2)
            deconv = lasagne.layers.Deconv2DLayer(input, output_units, ksize, crop='same', 
                                                  nonlinearity=nonlinearity)
            print deconv.output_shape
            return deconv
        
#         layer0 = decode_layer(reshape1, 1024, 512)
        layer1 = decode_layer(reshape1, 512, 256)
        layer2 = decode_layer(layer1, 256, 128)
        layer3 = decode_layer(layer2, 128, 64)
        layer4 = decode_layer(layer3, 64, 3, nonlinearity=lasagne.nonlinearities.tanh, depool=False)
        
        draft = lasagne.layers.dimshuffle(layer4, [0,2,3,1])
        output = draft
        print output.output_shape
        
        return output
    
    def Discriminate(self, T, F):
        print 'Initialling discriminator...'
        
        input_True_layer = lasagne.layers.InputLayer((None, 64, 64, 3), T)
        input_Fake_layer = lasagne.layers.InputLayer((None, 64, 64, 3), F)
        input_layer = ChooseLayer([input_True_layer, input_Fake_layer])
        dimshuffle1 = lasagne.layers.dimshuffle(input_layer, [0,3,1,2])
        
        def Discriminate_layer(input, num_units, ksize=(3,3)):
            input = lasagne.layers.BatchNormLayer(input)
            conv = lasagne.layers.Conv2DLayer(input, num_units, ksize, pad='same', nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2))
            pool = lasagne.layers.Conv2DLayer(conv, num_units, (2,2), stride=2, nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2))
            print pool.output_shape
            
            return pool
        
        layer1 = Discriminate_layer(dimshuffle1, num_units=32) #32*32
        layer2 = Discriminate_layer(layer1, num_units=64) #16*16
        layer3 = Discriminate_layer(layer2, num_units=128) #8*8
        layer4 = Discriminate_layer(layer3, num_units=256) #4*4
        layer5 = Discriminate_layer(layer4, num_units=256) #2*2
        
#         dense1 = lasagne.layers.DenseLayer(layer4, 1024, num_leading_axes=1)
        output = lasagne.layers.DenseLayer(layer5, 1, num_leading_axes=1,
                                           nonlinearity=None)
        
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
        
        self.y_hat = (lasagne.layers.get_output(self.decoder) + 1.) / 2. * 255.
        self.cost = self.Cost_l2(self.y_hat, self.y)
        self.f_cost = theano.function([self.x, self.y, self.mask], self.cost, name='Reconstruction cost function')
        
        self.generate = tensor.set_subtensor(self.x[:, 16:48, 16:48], self.y_hat)
        self.original = tensor.set_subtensor(self.x[:, 16:48, 16:48], self.y[:, 16:48, 16:48])
        
        self.disciminator = self.Discriminate(T=self.y, F=self.generate)
        true_score = lasagne.layers.get_output(self.disciminator, chs_idx=0)
        fake_score = lasagne.layers.get_output(self.disciminator, chs_idx=1)
        
        self.cost_Dsc = -(true_score - fake_score).mean() 
        self.f_cost_Dsc = theano.function([self.x, self.y, self.mask], self.cost_Dsc, name='GAN cost function')
        
        self.cost_Gen = - fake_score.mean()
        self.f_cost_Gen = theano.function([self.x, self.mask], self.cost_Gen, name='Generative cost function')
        
        self.test_cost = self.Cost_l2(self.y_hat, self.y[:, 16:48, 16:48])
        self.f_test_cost = theano.function([self.x, self.y, self.mask], self.test_cost, name='cost function')
        
        self.generator_params = lasagne.layers.get_all_params(self.decoder, trainable=True)
        self.discriminator_params = lasagne.layers.get_all_params(self.disciminator, trainable=True)
        
        self.f_yhat = theano.function([self.x, self.mask], self.y_hat, name='yhat')
        self.f_generate = theano.function([self.x, self.mask], self.generate, name='Generate')
        self.f_original = theano.function([self.x, self.y], self.original, name='Original')
        print 'Done initial'
        
    def show_examples(self, dataset):
        y_hat = self.f_yhat(numpy.array(dataset.valid_x[:10], dtype='float32'), dataset.mask)
        generate = self.f_generate(numpy.array(dataset.valid_x[:10], dtype='float32'), dataset.mask)
        original = self.f_original(numpy.array(dataset.valid_x[:10], dtype='float32'), 
                                   numpy.array(dataset.valid_y[:10], dtype='float32'))
        
        y_hat = y_hat.reshape([y_hat.shape[0] * y_hat.shape[1], y_hat.shape[2], y_hat.shape[3]])
        generate = generate.reshape([generate.shape[0] * generate.shape[1], generate.shape[2], generate.shape[3]])
        original = original.reshape([original.shape[0] * original.shape[1], original.shape[2], original.shape[3]])
        
        fig = numpy.int64(numpy.concatenate([original, generate], axis=1))
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
                    optimizer=lasagne.updates.rmsprop,
                    patience=5, 
                    lrate=0.00005, 
                    dispFreq=100, 
                    validFreq=-1, 
                    saveFreq=-1, 
                    max_epochs=40,
                    n_critic=5,
                    clip_params=0.01,
                    ):
        print 'Computing gradient...'
#         updates = optimizer(self.cost, self.generator_params, lrate)
#         self.f_update = theano.function([self.x, self.y, self.mask], self.cost, updates=updates)

        updates_Dsc = optimizer(self.cost_Dsc, self.discriminator_params, lrate=0.001)
        self.f_update_Dsc = theano.function([self.x, self.y, self.mask], self.cost_Dsc, updates=updates_Dsc)
        
        updates_Gen = optimizer(self.cost_Gen, self.generator_params, lrate)
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
                        
                        for p in self.discriminator_params:
                            pv = p.get_value()
                            pv = numpy.clip(pv, -clip_params, clip_params)
                            p.set_value(pv)
    
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
    
                    if numpy.mod(uidx, dispFreq) == 0:
                        nowtime = time.time()
                        print 'Epoch ', eidx, 'Update ', uidx - eidx * batchnum, '/', batchnum, 'Cost Gen', cost_Gen, 'Cost Dsc', cost_Dsc, \
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
        self.show_examples()
        return valid_decode_err
        
dataset = Mscoco('D:/workspace/Data/inpainting/')
model = Inpainting()
# model.learn_model(dataset, saveto='params/model.npz')
# model.load_model(saveto='params/model.npz')
# model.show_examples(dataset)
model.learn_model_GAN(dataset, saveto='params/model_GAN.npz')
