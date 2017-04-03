'''
Created on Feb 28, 2017

@author: Yikang
'''
import numpy
import theano.tensor as tensor

def softmax(x):
    e_x = tensor.exp(x - x.max(axis=-1, keepdims=True))
    out = e_x / e_x.sum(axis=-1, keepdims=True)
    
    return out

def get_minibatches_idx(n, minibatch_size, shuffle=True):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

#     if shuffle:
#         numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
        
    if shuffle:
        numpy.random.shuffle(minibatches)

    return zip(range(len(minibatches)), minibatches)

def error(f_cost, prepare_data, data_x, data_y, mask, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    total_words = 0
    for _, valid_index in iterator:
        x, y = prepare_data([data_x[t] for t in valid_index], [data_y[t] for t in valid_index])
        preds = f_cost(x, y, mask)
        valid_err += preds.sum()
        total_words += 1
    valid_err = valid_err / total_words

    return valid_err