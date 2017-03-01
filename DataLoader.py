'''
Created on Feb 28, 2017

@author: Yikang
'''

import os
import PIL.Image as Image
import numpy
import cPickle

class Mscoco(object):
    '''
    classdocs
    '''


    def __init__(self, path=''):
        '''
        Constructor
        '''
        self.path = path
        self.mask = numpy.zeros((64, 64, 3), dtype='uint8')
        self.mask[16:48, 16:48, :] = 1
        
        self.wdict = cPickle.load(open(path + 'worddict.pkl', 'rb'))
        captions = cPickle.load(open(path + 'dict_key_imgID_value_caps_train_and_valid.pkl', 'rb'))
        
        def load_dataset(path, captions):
            x = []
            y = []
            c = []
            files = os.listdir(path)
            n = 0
            for imgfile in files:
                img = Image.open(path + imgfile)
                img = numpy.array(img).astype('uint8')
                
                if img.shape != (64, 64, 3):
#                     print img.shape
                    continue
                
                x.append(img * (1 - self.mask))
                y.append(img[self.mask > 0].reshape((32,32,3)))
                c.append(captions[imgfile[:-4]])
                
                n += 1
                if n % 10000 == 0:
                    print n
#                     break
                
            return x, y, c
                
        print 'Loading data...'
        self.train_x, self.train_y, self.train_c = load_dataset(path + 'train2014/', captions)
        self.valid_x, self.valid_y, self.valid_c = load_dataset(path + 'val2014/', captions)
        print 'Done loading'
                