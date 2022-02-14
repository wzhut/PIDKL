"""
@author: Shandian Zhe
"""
#latent force models 

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import scipy.io
from scipy.interpolate import griddata
import time
from scipy.io import loadmat

np.random.seed(1234)
tf.set_random_seed(1234)

jitter = 1e-3


class GPR:
    #self.tf_log_lengthscale: log of RBF lengthscale
    #self.tf_log_tau: log of inverse variance
    #self.tf_X: training input
    #self.tf_y: training output
    #self.tf_Xt: test input

    #d is input dim.
    #heat equation
    #one latent force
    def __init__(self, X, y, t):
        # tf placeholders and graph
        self.X = X
        self.y = y
        self.t = t
        self.N,self.d = X.shape
        self.tf_X = tf.placeholder(tf.float32, shape=[None, self.d])
        self.tf_Xt = tf.placeholder(tf.float32, shape=[None, self.d])
        self.tf_y = tf.placeholder(tf.float32, shape=[None, 1])
        self.tf_t = tf.constant(self.t)
        #model parameters
        self.tf_log_v_0 = tf.Variable(0.0, dtype=tf.float32)
        self.tf_log_v = tf.Variable(0.0, dtype=tf.float32)
        self.tf_log_a = tf.Variable(0.0, dtype=tf.float32)
        #self.tf_lengthscale = 8*tf.exp(self.tf_log_a)*self.tf_t + tf.exp(self.tf_log_v)
        #self.tf_amp = tf.exp(self.tf_log_v_0)*tf.pow(tf.exp(self.tf_log_v)/(tf.exp(self.tf_log_v)+8*tf.exp(self.tf_log_a)*self.tf_t), 0.5*self.d)
        self.tf_log_lengthscale = tf.log(8*tf.exp(self.tf_log_a)*self.tf_t + tf.exp(self.tf_log_v))
        self.tf_log_amp = self.tf_log_v_0 + 0.5*self.d*(self.tf_log_v - self.tf_log_lengthscale)
        #self.tf_log_lengthscale = tf.Variable(0.0, dtype=tf.float32)
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
        #K = self.tf_amp*tf.exp(-1.0/self.tf_lengthscale*K)
        K = tf.exp(self.tf_log_amp)*tf.exp(-1.0/tf.exp(self.tf_log_lengthscale)*K)
        K = K + jitter*tf.eye(self.N)
        return K

    def kernel_cross(self):
        col_norm1 = tf.reshape(tf.reduce_sum(self.tf_Xt*self.tf_Xt, 1), [-1, 1])
        col_norm2 = tf.reshape(tf.reduce_sum(self.tf_X*self.tf_X, 1), [-1, 1])
        K = col_norm1 - 2.0*tf.matmul(self.tf_Xt, tf.transpose(self.tf_X)) + tf.transpose(col_norm2)
        #K = self.tf_amp*tf.exp(-1.0/self.tf_lengthscale*K)
        K = tf.exp(self.tf_log_amp)*tf.exp(-1.0/tf.exp(self.tf_log_lengthscale)*K)
        return K


    def pred(self):
        S = self.kernel_matrix() + 1.0/tf.exp(self.tf_log_tau)*tf.eye(self.N)
        Kmn = self.kernel_cross()
        pred_mean = tf.matmul(Kmn, tf.matrix_solve(S, self.tf_y))
        pred_var = tf.exp(self.tf_log_amp) + jitter - tf.reduce_sum(Kmn*tf.transpose(tf.matrix_solve(S, tf.transpose(Kmn))), 1)
        return (pred_mean, pred_var)
 
    def test(self, X_star):
        pred_mean,pred_var = self.pred()
        print (X_star.shape)
        print('tau = %g, length-scale = %g'%(np.exp(self.tf_log_tau.eval(session=self.sess)),  np.exp(self.tf_log_lengthscale.eval(session=self.sess))))
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
        print('a = %g'%np.exp(self.tf_log_a.eval(session=self.sess)))



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
    e = []
    l = []
    for nfold in range(1, 6):
        infile = 'f' + str(nfold) + '.mat'
        data = loadmat(infile, squeeze_me=True, struct_as_record=False, mat_dtype=True)

        X_train = data['X_train']
        X_test = data['X_test']
        Y_train = data['Y_train'].reshape((-1, 1))
        Y_test = data['Y_test'].reshape((-1, 1))
        # normalization
        y_mean = np.mean(Y_train)
        y_std = np.std(Y_train)

        x_mean = np.mean(X_train, axis=0)
        x_std = np.std(X_train, axis=0)

        x_std[x_std==0] = 1

        Y_train = (Y_train - y_mean) / y_std
        Y_test = (Y_test - y_mean) / y_std

        X_train = (X_train - x_mean) / x_std
        X_test = (X_test - x_mean) / x_std

        model = GPR(X_train, Y_train, 1.0)
        model.train()
        y_pred = model.test(X_test)
        y_mean = y_pred[0]
        y_var = y_pred[1]
        error = np.linalg.norm(y_mean - Y_test, 2)/np.linalg.norm(Y_test, 2)
        ll = -0.5 * np.log(2 * np.pi * y_var) - 0.5 * (Y_test - y_mean)**2 / y_var
        ll = np.mean(ll)
        e.append(error)
        l.append(ll)
        print('Error u: %e' % (error))                     
    #print(y_var)
    print(e)
    print(l)


    
