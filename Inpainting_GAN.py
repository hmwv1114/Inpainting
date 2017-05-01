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
from Layers import ApplyMask, ChooseLayer, ApplyNoise

from Inpainting import Inpainting

rng = numpy.random.RandomState(seed=123456)

class Inpainting_GAN(Inpainting):
    '''
    classdocs
    '''
        
    def init_model(self):
        print 'Initialing model'
        self.x = tensor.tensor4(name='x', dtype='float32')
        self.y = tensor.tensor4(name='y', dtype='float32')
        self.mask = tensor.matrix(name='mask', dtype='float32')
        
        self.c = tensor.matrix(name='c', dtype='int64')
        self.cmask = tensor.matrix(name='cmask', dtype='float32')
        
        self.c_encoder = self.Cap_Encode(self.c, self.cmask)
        
        self.encoder = self.Encode(self.x, self.mask, self.c_encoder)
        
        self.decoder = self.Decode(self.encoder)
        
        self.y_hat = lasagne.layers.get_output(self.decoder, 
                                               batch_norm_update_averages=True)
        self.cost = self.Cost_l2(self.y_hat, self.y[:, 16:48, 16:48])
        
        self.generate = tensor.set_subtensor(self.x[:, 16:48, 16:48], self.y_hat)
        self.original = tensor.set_subtensor(self.x[:, 16:48, 16:48], self.y[:, 16:48, 16:48])
        
        self.disciminator = self.Discriminate(self.y)

        real_score = lasagne.layers.get_output(self.disciminator, 
                                               batch_norm_update_averages=True)
        fake_score = lasagne.layers.get_output(self.disciminator, self.generate,
                                               batch_norm_update_averages=True)
        
        self.cost_Dsc = (lasagne.objectives.binary_crossentropy(real_score, 1) 
                         + lasagne.objectives.binary_crossentropy(fake_score, 0)
                         ).mean()
        self.f_cost_Dsc = theano.function([self.x, self.y, self.mask, self.c, self.cmask], self.cost_Dsc, 
                                          name='Discriminative cost function')
        
        self.cost_Gen = lasagne.objectives.binary_crossentropy(fake_score, 1).mean()
        self.cost_Gen = 0.5 * self.cost_Gen + 0.5 * self.cost
        self.f_cost_Gen = theano.function([self.x, self.mask, self.y, self.c, self.cmask], self.cost_Gen, 
                                          name='Generative cost function')
        
        self.test_cost = self.Cost_l2(self.y_hat, self.y[:, 16:48, 16:48])
        self.f_test_cost = theano.function([self.x, self.y, self.mask, self.c, self.cmask], self.test_cost, name='cost function')
        
        self.generator_params = lasagne.layers.get_all_params(self.decoder, trainable=True)
        print len(self.generator_params)
        self.discriminator_params = lasagne.layers.get_all_params(self.disciminator, trainable=True)
        print len(self.discriminator_params)
        
        self.y_hat_output = lasagne.layers.get_output(self.decoder, batch_norm_update_averages=True)
        self.generate_output = tensor.set_subtensor(self.x[:, 16:48, 16:48], self.y_hat_output)
        self.f_yhat = theano.function([self.x, self.mask, self.c, self.cmask], self.y_hat_output, name='yhat')
        self.f_generate = theano.function([self.x, self.mask, self.c, self.cmask], self.generate_output, name='Generate')
        self.f_original = theano.function([self.x, self.y], self.original, name='Original')
        print 'Done initial'
        
    def learn_model_GAN(self, dataset, 
                    batch_size=128, 
                    valid_batch_size=128, 
                    saveto='params/model', 
                    optimizer=lasagne.updates.adam,
                    patience=40, 
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
        self.f_update = theano.function([self.x, self.y, self.mask, self.c, self.cmask], self.cost, updates=updates)

        updates_Dsc = optimizer(self.cost_Dsc, self.discriminator_params, lrate, beta1=0.5)
        self.f_update_Dsc = theano.function([self.x, self.y, self.mask, self.c, self.cmask], self.cost_Dsc, updates=updates_Dsc)
        
        updates_Gen = optimizer(self.cost_Gen, self.generator_params, lrate, beta1=0.5)
        self.f_update_Gen = theano.function([self.x, self.mask, self.y, self.c, self.cmask], self.cost_Gen, updates=updates_Gen)
            
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
                        c = [dataset.train_c[t]for t in train_index_sample]
                        y0 = [dataset.train_y[t]for t in train_index_sample]
                        x0, y0, c, cmask = dataset.prepare_data(x, y0, c)
                        cost_Dsc = self.f_update_Dsc(x0, y0, dataset.mask, c, cmask)
                        
                        random_idx = rng.randint(len(kf))
                        _, train_index_sample = kf[random_idx]
                        y1 = [dataset.train_y[t]for t in train_index_sample]
                        
                        if len(x) != batch_size or len(y1) != batch_size:
                            print 'unmatched sample'
                            continue
        
                        # Get the data in numpy.ndarray format
                        # This swap the axis!
                        # Return something of shape (minibatch maxlen, n samples)
                        x1, y1, c, cmask = dataset.prepare_data(x, y1, c)
        
                        cost_Dsc = self.f_update_Dsc(x1, y1, dataset.mask, c, cmask)
                        
                        if numpy.isnan(cost_Dsc) or numpy.isinf(cost_Dsc):
                            print 'bad cost detected: ', cost_Dsc
                            return 1., 1., 1.
    
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
                     
                    cost_Gen = self.f_update_Gen(x, dataset.mask, y, c, cmask)
    
                    if numpy.isnan(cost_Gen) or numpy.isinf(cost_Gen):
                        print 'bad cost detected: ', cost_Gen
                        return 1., 1., 1.

#                     cost = self.f_update(x, y ,dataset.mask)
                    cost = -1
    
                    if numpy.mod(uidx, dispFreq) == 0:
                        nowtime = time.time()
                        print 'Epoch ', eidx, 'Update ', uidx - eidx * batchnum, '/', batchnum, 'Cost Gen', cost_Gen, 'Cost Dsc', cost_Dsc, 'Cost L2', cost, \
                              'Time cost ', nowtime - start_time, 'Expected epoch time cost ', (nowtime - start_time) * batchnum / uidx
    
                    if numpy.mod(uidx, validFreq) == 0:
                        #train_err = pred_error(f_decode, prepare_data, train, kf)
                        valid_decode_err = error(self.f_cost_Dsc, dataset.prepare_data, dataset.valid_x, dataset.valid_y, dataset.valid_c, dataset.mask, kf_valid)
    
                        self.history_errs.append(valid_decode_err)
    
                        if (self.best_p is None 
                            or 
                            valid_decode_err <= numpy.array(self.history_errs).min()):
    
                            del self.best_p
                            self.best_p = lasagne.layers.get_all_param_values(self.decoder)
                            bad_counter = 0
                            
                            self.save_model(saveto)
                            
                        self.show_examples(dataset, batch_size, 'figures/GAN_epoch' + str(eidx) + '.png')
    
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

if __name__ == '__main__':
    dataset = Mscoco('../Data/inpainting/')
    model = Inpainting_GAN(max(dataset.wdict.values()) + 1)
    # model.learn_model(dataset, saveto='params/model.npz')
    # model.load_model(saveto='params/model.npz')
    # model.show_examples(dataset)
    model.learn_model_GAN(dataset, saveto='params/model_GAN.npz')
