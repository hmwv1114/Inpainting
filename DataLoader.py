'''
Created on Feb 28, 2017

@author: Yikang
'''

import os
import PIL.Image as Image
import numpy
import cPickle
from nltk.tokenize import WordPunctTokenizer

rng = numpy.random.RandomState(seed=123456)

class Mscoco(object):
    '''
    classdocs
    '''


    def __init__(self, path=''):
        '''
        Constructor
        '''
        self.path = path
        self.mask = numpy.ones((64, 64), dtype='float32')
        self.mask[16:48, 16:48] = 0
        
        self.wdict = cPickle.load(open(path + 'worddict.pkl', 'rb'))
        captions = cPickle.load(open(path + 'dict_key_imgID_value_caps_train_and_valid.pkl', 'rb'))
        
        def load_dataset(path, captions):
            x = []
            y = []
            c = []
            files = os.listdir(path)
            n = 0
            tokenizer = WordPunctTokenizer()
            for imgfile in files:
                img = Image.open(path + imgfile)
                img = numpy.array(img).astype('uint8')
                
                if img.shape != (64, 64, 3):
#                     print img.shape
                    continue
                
                x.append(img * self.mask[:,:,None])
#                 y.append(img[self.mask == 0].reshape((32,32,3)))
                y.append(img)
                cap = captions[imgfile[:-4]]
                cap = ' '.join(cap).lower()
#                 print cap
                cap = tokenizer.tokenize(cap)
                seq = []
                for w in cap:
                    if self.wdict.has_key(w):
                        seq.append(self.wdict[w])
                c.append(seq)
                
                n += 1
                if n % 10000 == 0:
                    print n
#                     break
                
            return x, y, c
                
        print 'Loading data...'
        self.train_x, self.train_y, self.train_c = load_dataset(path + 'train2014/', captions)
        self.valid_x, self.valid_y, self.valid_c = load_dataset(path + 'val2014/', captions)
        print 'Done loading'
        
    def prepare_data(self, x, y, seqs):
        x = numpy.array(x, dtype='float32') / 255. * 2. - 1.
        y = numpy.array(y, dtype='float32') / 255. * 2. - 1.
#         x = x * (1 - self.mask) + rng.randint(0, 256, x.shape) * self.mask

        lengths = [len(s) for s in seqs]

        if len(lengths) < 1:
            return None, None, None
    
        n_samples = len(seqs)
        maxlen = numpy.max(lengths)
    
        c = numpy.zeros((n_samples, maxlen)).astype('int64')
        cmask = numpy.zeros((n_samples, maxlen)).astype('float32')
        for idx, s in enumerate(seqs):
            c[idx, :lengths[idx]] = s
            cmask[idx, :lengths[idx]] = 1.

        return x.astype('float32'), y.astype('float32'), c.astype('int64'), cmask.astype('float32')
                