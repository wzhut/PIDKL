"""
@author: Shandian Zhe
"""

#test ode data grom Wei

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time

np.random.seed(1234)
tf.set_random_seed(1234)

jitter = 1e-5

class GPR:
    #self.tf_log_lengthscale: log of RBF lengthscale
    #self.tf_log_tau: log of inverse variance
    #self.tf_X: training input
    #self.tf_y: training output
    #self.tf_Xt: test input

#d is input dim.
    def __init__(self, X, y):
        # tf placeholders and graph
        self.X = X
        self.y = y
        self.N,self.d = X.shape
        self.tf_X = tf.placeholder(tf.float32, shape=[None, self.d])
        self.tf_y = tf.placeholder(tf.float32, shape=[None, 1])
        self.tf_Xt = tf.placeholder(tf.float32, shape=[None, self.d])
        #model parameters
        self.tf_log_lengthscale = tf.Variable(0.0, dtype=tf.float32)
        #self.tf_log_amp = tf.Variable(0.0, dtype=tf.float32)
        self.tf_log_amp = tf.constant(0.0, dtype=tf.float32)
        self.tf_log_tau = tf.Variable(0.0, dtype=tf.float32)
        self.loss = self.neg_log_evidence()
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        self.sess = tf.Session() 
        self.sess.run(tf.global_variables_initializer())
        
    def kernel_matrix(self):
        #rbf kernel
        col_norm2 = tf.reduce_sum(self.tf_X*self.tf_X, 1)
        col_norm2 = tf.reshape(col_norm2, [-1,1])
        K = col_norm2 - 2.0*tf.matmul(self.tf_X, tf.transpose(self.tf_X)) + tf.transpose(col_norm2)
        K = tf.exp(self.tf_log_amp)*tf.exp(-1.0/tf.exp(self.tf_log_lengthscale)*K)
        K = K + jitter*tf.eye(self.N)
        return K

    def kernel_cross(self):
        col_norm1 = tf.reshape(tf.reduce_sum(self.tf_Xt*self.tf_Xt, 1), [-1, 1])
        col_norm2 = tf.reshape(tf.reduce_sum(self.tf_X*self.tf_X, 1), [-1, 1])
        K = col_norm1 - 2.0*tf.matmul(self.tf_Xt, tf.transpose(self.tf_X)) + tf.transpose(col_norm2)
        K = tf.exp(self.tf_log_amp)*tf.exp(-1.0/tf.exp(self.tf_log_lengthscale)*K)
        return K


    def pred(self):
        S = self.kernel_matrix() + 1.0/tf.exp(self.tf_log_tau)*tf.eye(self.N)
        Kmn = self.kernel_cross()
        pred_mean = tf.matmul(Kmn, tf.matrix_solve(S, self.tf_y))
        pred_var = tf.exp(self.tf_log_amp) + jitter - tf.reduce_sum(Kmn*tf.transpose(tf.matrix_solve(S, tf.transpose(Kmn))), 1)
        pred_var = tf.reshape(pred_var,[-1,1])
        return (pred_mean, pred_var)
 
    def test(self, X_star):
        pred_mean,pred_var = self.pred()
        print (X_star.shape)
        print('tau = %g, amp = %g, length-scale = %g'%(np.exp(self.tf_log_tau.eval(session=self.sess)), np.exp(self.tf_log_amp.eval(session=self.sess)), np.exp(self.tf_log_lengthscale.eval(session=self.sess))))
        return  self.sess.run([pred_mean, pred_var], {self.tf_Xt: X_star, self.tf_X: self.X, self.tf_y:self.y})

    #self.X, self.y
    def neg_log_evidence(self):
        S = self.kernel_matrix() + 1.0/tf.exp(self.tf_log_tau)*tf.eye(self.N)
        L = 0.5*tf.linalg.logdet(S) + 0.5*tf.matmul(tf.transpose(self.tf_y), tf.matrix_solve(S, self.tf_y))[0,0]
        return L
       
    def callback(self, loss):
        print('Loss:', loss)

    def train(self):
        tf_dict = {self.tf_X:self.X, self.tf_y:self.y}
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        

        print('tau = %g, length-scale = %g'%(np.exp(self.tf_log_tau.eval(session=self.sess)),  np.exp(self.tf_log_lengthscale.eval(session=self.sess))))



#sample training & test
def sample_data():
    n_total = 200
    f = lambda x: np.sin(x)
    x = np.random.rand(n_total, 1)
    y = f(x) + 0.1*np.random.randn(n_total,1)
    return (x,y)
    

def test_GPR():
    X,y = sample_data()
    n_tr = 100
    X_tr = X[1:n_tr,:]
    y_tr = y[1:n_tr]
    X_test = X[n_tr:,:]
    y_test = y[n_tr:]
    
    model = GPR(X_tr, y_tr)
    model.train()
    y_pred = model.test(X_test)
    y_mean = y_pred[0]
    y_var = y_pred[1]
    y_test = np.sin(X_test)
    error = np.linalg.norm(y_mean - y_test, 2)/np.linalg.norm(y_test, 2)
    print('Error u: %e' % (error))                     
    print(y_var)


def to_vec(x):
    return x.reshape([x.size,1])
        
if __name__ == "__main__": 
    data = scipy.io.loadmat('./data_ode.mat')
    train = data['data']['train'][0][0]
    test = data['data']['test'][0][0]
    Xtr = (train[:,0:-1])
    ytr = to_vec(train[:,-1])
    X_test = (test[:,0:-1])
    y_test = to_vec(test[:,-1])

    #extrapolation
    Xtr = X_test[0:101,:]
    X_test = X_test[101:,:]
    ytr = y_test[0:101,:]
    y_test = y_test[101:,:]
    print(Xtr.shape)

    model = GPR(Xtr, ytr)
    model.train()
    y_pred_tr = model.test(Xtr)
    y_mean_tr = y_pred_tr[0]
    y_var_tr = y_pred_tr[1]

    y_pred = model.test(X_test)
    y_mean = y_pred[0]
    y_var = y_pred[1]
    rmse = np.linalg.norm(y_mean - y_test, 2)
    nrmse = rmse/np.linalg.norm(y_test, 2)
    res = {'rmse':rmse, 'nrmse':nrmse, 'xtr':Xtr, 'ytr':ytr, 'pred_mean_tr':y_mean_tr, 'pred_var_tr':y_var_tr, 'xtest':X_test, 'y_test':y_test, 'pred_mean':y_mean, 'pred_var':y_var }
    print('rmse = %g, nrmse = %g' % (rmse, nrmse))                     
    scipy.io.savemat('res_GPR.mat',res)
    #print(y_var)


    
