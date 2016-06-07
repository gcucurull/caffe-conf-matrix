import caffe
import json
import numpy as np
import sys

import sklearn.metrics

class PythonConfMat(caffe.Layer):
    """
    Compute the Accuracy with a Python Layer
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs.")

        self.num_labels = bottom[0].channels
        params = json.loads(self.param_str)
        self.test_iter = params['test_iter']
        self.conf_matrix = np.zeros((self.num_labels, self.num_labels))
        self.current_iter = 0

    def reshape(self, bottom, top):
        # bottom[0] are the net's outputs
        # bottom[1] are the ground truth labels

        # Net outputs and labels must have the same number of elements
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same number of elements.")

        # accuracy output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.current_iter += 1

        # predicted outputs
        pred = np.argmax(bottom[0].data, axis=1)
        accuracy = np.sum(pred == bottom[1].data).astype(np.float32) / bottom[0].num
        top[0].data[...] = accuracy

        # compute confusion matrix
        self.conf_matrix += sklearn.metrics.confusion_matrix(bottom[1].data, pred, labels=range(self.num_labels))
        
        if self.current_iter == self.test_iter:
            self.current_iter = 0
            sys.stdout.write('\nCAUTION!! test_iter = %i. Make sure this is the correct value' % self.test_iter)
            sys.stdout.write('\n"param_str: \'{"test_iter":%i}\'" has been set in the definition of the PythonLayer' % self.test_iter)
            sys.stdout.write('\n\nConfusion Matrix')
            sys.stdout.write('\t'*(self.num_labels-2)+'| Accuracy')
            sys.stdout.write('\n'+'-'*8*(self.num_labels+1))
            sys.stdout.write('\n')
            for i in range(len(self.conf_matrix)):
                for j in range(len(self.conf_matrix[i])):
                    sys.stdout.write(str(self.conf_matrix[i][j].astype(np.int))+'\t')
                sys.stdout.write('| %3.2f %%' % (self.conf_matrix[i][i]*100 / self.conf_matrix[i].sum()))
                sys.stdout.write('\n')
            sys.stdout.write('Number of test samples: %i \n\n' % self.conf_matrix.sum())
            # reset conf_matrix for next test phase
            self.conf_matrix = np.zeros((self.num_labels, self.num_labels))

    def backward(self, top, propagate_down, bottom):
        pass
