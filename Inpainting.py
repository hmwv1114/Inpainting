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
        print('initial model...')
        self.init_model()
        print('load parameters')
        self.load_params(saveto)
        print('Done')
        
    def Encode(self, x):
        input_layer = lasagne.layers.InputLayer((None, 64, 64, 3), x/255.)
        dimshuffle1 = lasagne.layers.dimshuffle(input_layer, [0,3,1,2])
        conv1 = lasagne.layers.Conv2DLayer(dimshuffle1, 12, (3,3), pad='same')
#         print conv1.output_shape
        pool1 = lasagne.layers.Pool2DLayer(conv1, (2,2))
        print pool1.output_shape
        conv2 = lasagne.layers.Conv2DLayer(pool1, 48, (3,3), pad='same')
#         print conv2.output_shape
        pool2 = lasagne.layers.Pool2DLayer(conv2, (2,2))
        print pool2.output_shape
        conv3 = lasagne.layers.Conv2DLayer(pool2, 192, (3,3), pad='same')
#         print conv3.output_shape
        pool3 = lasagne.layers.Pool2DLayer(conv3, (2,2))
        print pool3.output_shape
        conv4 = lasagne.layers.Conv2DLayer(pool3, 768, (3,3), pad='same')
        pool4 = lasagne.layers.Pool2DLayer(conv4, (2,2))
        print pool4.output_shape
#         conv5 = lasagne.layers.Conv2DLayer(pool4, 768, (3,3), pad='same')
#         pool5 = lasagne.layers.Pool2DLayer(conv5, (2,2))
#         print pool5.output_shape
        
        return pool4
    
    def Decode(self, encoder):
        channel_full1 = lasagne.layers.DenseLayer(encoder, 16, num_leading_axes=2)
        reshape1 = lasagne.layers.reshape(channel_full1, ([0], [1], 4, 4))
        print reshape1.output_shape
        depool2 = lasagne.layers.Deconv2DLayer(reshape1, 768, (2, 2), stride=2)
        deconv2 = lasagne.layers.Deconv2DLayer(depool2, 192, (3, 3), crop='same')
        print deconv2.output_shape
        depool3 = lasagne.layers.Deconv2DLayer(deconv2, 192, (2, 2), stride=2)
        deconv3 = lasagne.layers.Deconv2DLayer(depool3, 48, (3, 3), crop='same')
        print deconv3.output_shape
        depool4 = lasagne.layers.Deconv2DLayer(deconv3, 48, (2, 2), stride=2)
        deconv4 = lasagne.layers.Deconv2DLayer(depool4, 12, (3, 3), crop='same')
        print deconv4.output_shape
        depool5 = lasagne.layers.Deconv2DLayer(deconv4, 12, (2, 2), stride=2)
        deconv5 = lasagne.layers.Deconv2DLayer(depool5, 3, (3, 3), crop='same', 
                                               nonlinearity=None)
        print deconv5.output_shape
        output = lasagne.layers.dimshuffle(deconv5, [0,2,3,1])
        print output.output_shape
        
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
        self.y = tensor.tensor4(name='y', dtype='int64')
        
        self.encoder = self.Encode(self.x)
        
        self.decoder = self.Decode(self.encoder)
        
        self.y_hat = lasagne.layers.get_output(self.decoder)
        
        self.cost = self.Cost_l2(self.y_hat, self.y)
        self.f_cost = theano.function([self.x, self.y], self.cost, name='cost function')
        
        self.test_cost = self.Cost_l2(self.y_hat[:, 16:48, 16:48], self.y[:, 16:48, 16:48])
        self.f_test_cost = theano.function([self.x, self.y], self.test_cost, name='cost function')
        
        self.trainable_params = lasagne.layers.get_all_params(self.decoder, trainable=True)
        
        self.generate = tensor.set_subtensor(self.x[:, 16:48, 16:48], self.y_hat[:, 16:48, 16:48])
        self.original = tensor.set_subtensor(self.x[:, 16:48, 16:48], self.y[:, 16:48, 16:48].astype('float32'))
        self.f_yhat = theano.function([self.x], self.y_hat, name='yhat')
        self.f_generate = theano.function([self.x], self.generate, name='Generate')
        self.f_original = theano.function([self.x, self.y], self.original, name='Original')
        print 'Done initial'
        
    def show_examples(self):
        y_hat = self.f_yhat(numpy.array(dataset.valid_x[:10], dtype='float32'))
        generate = self.f_generate(numpy.array(dataset.valid_x[:10], dtype='float32'))
        original = self.f_original(numpy.array(dataset.valid_x[:10], dtype='float32'), 
                                   numpy.array(dataset.valid_y[:10], dtype='int64'))
        
        y_hat = y_hat.reshape([y_hat.shape[0] * y_hat.shape[1], y_hat.shape[2], y_hat.shape[3]])
        generate = generate.reshape([generate.shape[0] * generate.shape[1], generate.shape[2], generate.shape[3]])
        original = original.reshape([original.shape[0] * original.shape[1], original.shape[2], original.shape[3]])
        
        fig = numpy.int64(numpy.concatenate([original, generate, y_hat], axis=1))
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
                    max_epochs=40):
        print 'Computing gradient...'
        if lrate is None:
            updates = optimizer(self.cost, self.trainable_params)
        else:
            updates = optimizer(self.cost, self.trainable_params, lrate)
        self.f_update = theano.function([self.x, self.y], self.cost, updates=updates)
            
        print('Optimization')
        kf_valid = get_minibatches_idx(len(dataset.valid_x), valid_batch_size)
    
        print("%d train examples" % len(dataset.train_x))
        print("%d valid examples" % len(dataset.valid_x))
    
        bad_count = 0
    
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
    
                    cost = self.f_update(x, y)
    
                    if numpy.isnan(cost) or numpy.isinf(cost):
                        print 'bad cost detected: ', cost
                        return 1., 1., 1.
    
                    if numpy.mod(uidx, dispFreq) == 0:
                        nowtime = time.time()
                        print 'Epoch ', eidx, 'Update ', uidx - eidx * batchnum, '/', batchnum, 'Cost ', cost, \
                              'Time cost ', nowtime - start_time, 'Expected epoch time cost ', (nowtime - start_time) * batchnum / uidx
    
                    if numpy.mod(uidx, validFreq) == 0:
                        #train_err = pred_error(f_decode, prepare_data, train, kf)
                        valid_decode_err = error(self.f_test_cost, dataset.prepare_data, dataset.valid_x, dataset.valid_y, kf_valid)
    
                        self.history_errs.append(valid_decode_err)
    
                        if (self.best_p is None 
                            or 
                            valid_decode_err <= numpy.array(self.history_errs).min()):
    
                            del self.best_p
                            self.best_p = lasagne.layers.get_all_param_values(self.decoder)
                            bad_counter = 0
                            
                            self.save_model(saveto)
                            
                            self.show_examples()
    
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
        valid_decode_err = error(self.f_test_cost, dataset.prepare_data, dataset.valid_x, dataset.valid_y, kf_valid)
    
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
model.learn_model(dataset)
