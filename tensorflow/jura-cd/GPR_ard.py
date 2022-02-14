"""
@author: Zheng Wang
"""

#use random samples and ADAM
#use batch size = 100

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import time
import scipy
import ExpUtil as util


jitter = 1e-3
tf_type = util.tf_type

class GPR:

    def __init__(self, cfg):
        self.cfg = cfg
        # tf placeholders and graph
        
        self.layers = cfg['layers']

        self.input_dim = cfg['input_dim']
        self.output_dim = cfg['output_dim']


        self.tf_X_f = tf.placeholder(tf_type,shape=[None, self.input_dim])
        self.tf_y = tf.placeholder(tf_type, shape=[None, self.output_dim])


        # Initialize NNs
        self.kernel_ard = util.Kernel_ARD(jitter) 

        #model parameters
        # f kernel
        self.tf_log_ls = tf.Variable(0.0, dtype=tf_type)
        self.tf_log_amp = tf.Variable(0.0, dtype=tf_type)
        # noise level
        self.tf_log_tau = tf.Variable(0.0, dtype=tf_type)

        self.nELBO = self.get_nELBO()
        # testing
        self.tf_X_test = tf.placeholder(tf_type,shape=[None, self.input_dim])
        self.tf_y_test = self.pred(self.tf_X_test)

        self.optimizer = tf.train.AdamOptimizer(cfg['lr'])
        self.minimizer = self.optimizer.minimize(self.nELBO)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def mul_trace(self, X1, X2):
        return tf.reduce_sum(X1 * tf.transpose(X2))

    def get_nELBO(self):
        n_X = (self.tf_X_f)
        Knn = self.kernel_ard.matrix(n_X, tf.exp(self.tf_log_amp), tf.exp(self.tf_log_ls))
        N = tf.cast(tf.shape(self.tf_X_f)[0], dtype=tf_type)
        # lpy
        S = Knn + 1.0 / tf.exp(self.tf_log_tau) * tf.eye(N)
        lpy = -0.5 * tf.linalg.logdet(S) - 0.5 * tf.matmul(tf.transpose(self.tf_y), tf.linalg.solve(S, self.tf_y))
        nELBO = -lpy
        return nELBO
    
    
    def pred(self, X):
        NN_X_f = (self.tf_X_f)
        Knn = self.kernel_ard.matrix(NN_X_f, tf.exp(self.tf_log_amp), tf.exp(self.tf_log_ls))
        N = tf.shape(self.tf_X_f)[0]
        S = Knn + 1.0 / tf.exp(self.tf_log_tau) * tf.eye(N)
        
        NN_X = (X)
        Kmn = self.kernel_ard.cross(NN_X, NN_X_f, tf.exp(self.tf_log_amp), tf.exp(self.tf_log_ls))
        
        post_mean = tf.matmul(Kmn, tf.linalg.solve(S, self.tf_y))
        # diagonal
        post_var = tf.exp(self.tf_log_amp) + jitter - tf.reduce_sum(Kmn * tf.transpose(tf.linalg.solve(S, tf.transpose(Kmn))), 1)
        post_std = tf.reshape(tf.sqrt(post_var), tf.shape(post_mean))

        return post_mean, post_std 
    

    def train(self, ofile):
        nepoch = self.cfg['nepoch']
        nepoch = self.cfg['nepoch']
        X_train = self.cfg['X_train'].reshape((-1, self.input_dim))
        y_train = self.cfg['Y_train'].reshape((-1, self.output_dim))
        X_test = self.cfg['X_test'].reshape((-1, self.input_dim))
        y_test = self.cfg['Y_test'].reshape((-1, self.output_dim))
        min_error = 1
        max_ll = -10000
        for i in range(nepoch):
            tf_dict = {self.tf_X_f: X_train, self.tf_y: y_train, self.tf_X_test: X_test}
            self.sess.run(self.minimizer,feed_dict = tf_dict)
            if i%100==0:
                y_mean, y_std = self.sess.run(self.tf_y_test, feed_dict=tf_dict)
                y_var = y_std ** 2
                error_f = np.linalg.norm(y_test- y_mean,2)/np.linalg.norm(y_test,2)
                ll = -0.5 * np.log(2 * np.pi * y_var) - 0.5 * (y_test - y_mean)**2 / y_var
                ll = np.mean(ll)
                if error_f < min_error:
                    min_error = error_f
                if ll > max_ll:
                    max_ll = ll
                np.savetxt(ofile, [min_error, max_ll])
                print('epoch %d, Error f: %e, LL: %e' % (i, error_f, ll))
                with open('result.txt','a+') as f:
                    f.write('epoch %d, Error u: %e\n' % (i, error_f))
        print('tau = %g, length-scale = %g'%(np.exp(self.tf_log_tau.eval(session=self.sess)),  np.exp(self.tf_log_ls.eval(session=self.sess))))
        print('min_error:', min_error)
        print('max_ll:', max_ll)
        return min_error, max_ll





