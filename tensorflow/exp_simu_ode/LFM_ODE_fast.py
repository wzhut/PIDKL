"""
@author: Shandian Zhe
"""

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
#from pyDOE import lhs
import time

np.random.seed(1234)
tf.set_random_seed(1234)

jitter = 1e-5

#ODE 
class LFM:
    #self.tf_log_lengthscale: log of RBF lengthscale
    #self.tf_log_tau: log of inverse variance
    #self.tf_X: training input
    #self.tf_y: training output
    #self.tf_Xt: test input

    def __init__(self, X, y, R):
        # tf placeholders and graph
        self.X = X
        self.y = y
        self.R = R
        self.N,self.d = X.shape
        self.tf_X = tf.placeholder(tf.float32, shape=[None, self.d])
        self.tf_y = tf.placeholder(tf.float32, shape=[None, 1])
        #this is test input
        self.tf_Xt = tf.placeholder(tf.float32, shape=[None, self.d])
        #noise for model parameters
        self.tf_log_tau = tf.Variable(0.0, dtype=tf.float32)
        self.tf_log_s0 = tf.Variable(0.0, dtype=tf.float32)
        #for R latent forces
        self.tf_log_lr2 = tf.Variable(tf.zeros([self.R]), dtype=tf.float32)
        self.tf_log_s = tf.Variable(tf.zeros([self.R]), dtype=tf.float32)
        #no need to learn B here, not in covariance
        #self.tf_log_B = tf.Variable([0.0], dtype=tf.float32)
        self.tf_log_D = tf.Variable([0.0], dtype=tf.float32)
        self.tf_D = tf.exp(self.tf_log_D)
        self.v_r = [None for r in range(self.R)]
        self.lr = [None for r in range(self.R)]
        for r in range(self.R):
            self.lr[r] = tf.exp(0.5*self.tf_log_lr2[r])
            self.v_r[r] = 0.5*self.lr[r]*self.tf_D

        #S = self.kernel_matrix(self.k_uu, self.tf_X) + 1.0/tf.exp(self.tf_log_tau)*tf.eye(self.N)
        S = self.kernel_matrix2(self.tf_X) + 1.0/tf.exp(self.tf_log_tau)*tf.eye(self.N)
        self.S = S
        negL = 0.5*self.N*np.log(2.0*np.pi) + 0.5*tf.linalg.logdet(S) + 0.5*tf.matmul(tf.transpose(self.tf_y), tf.matrix_solve(S, self.tf_y))[0,0]
        self.loss = negL 
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 5000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 100,
                                                                           'maxls': 100,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.minimizer = self.optimizer.minimize(self.loss)
        self.sess = tf.Session() 
        self.sess.run(tf.global_variables_initializer())


    #define your own kernel here
    def _ard_sq(self, x, y):
        return self.tf_amplitude*tf.exp(-0.5*tf.reduce_sum((x-y)/self.tf_lengthscale*(x-y)))

    def _h(self, t_p, t, r):
        res = tf.exp(tf.pow(self.v_r[r],2))/(2*self.tf_D)*tf.exp(-self.tf_D*t_p)*( \
        tf.exp(self.tf_D*t)*(tf.erf((t_p - t)/self.lr[r] - self.v_r[r]) + tf.erf(t/self.lr[r] + self.v_r[r]))  \
        - tf.exp(- self.tf_D*t)*(tf.erf(t_p/self.lr[r] - self.v_r[r]) + tf.erf(self.v_r[r])))
        return res

    def k_uu(self, t, t_p):
        res = 0
        for r in range(self.R):
            res = res + 0.5*np.sqrt(np.pi)*tf.exp(self.tf_log_s[r]*2)*self.lr[r]*(self._h(t_p, t, r) + self._h(t, t_p, r))
        return res

    #direct compute
    #X: N by 1, Y: M by 1
    def kernel_cross2(self, X, Y):
        N = tf.shape(X)[0]
        M = tf.shape(Y)[0]
        K = tf.zeros([N, M])
        for r in range(self.R):
            Xdiff = X - tf.transpose(Y)
            Xsum = X + tf.transpose(Y)
            H = tf.exp(-self.tf_D*Xdiff)*(tf.erf(Xdiff/self.lr[r] - self.v_r[r]) + tf.erf(tf.transpose(Y)/self.lr[r] + self.v_r[r]) ) - tf.exp(-self.tf_D*Xsum)*(tf.erf(X/self.lr[r] - self.v_r[r]) + tf.erf(self.v_r[r]))
            H = tf.exp(tf.pow(self.v_r[r],2))/(2*self.tf_D)*H 

            Xdiff = Y - tf.transpose(X)
            Xsum = Y + tf.transpose(X)
            G = tf.exp(-self.tf_D*Xdiff)*(tf.erf(Xdiff/self.lr[r] - self.v_r[r]) + tf.erf(tf.transpose(X)/self.lr[r] + self.v_r[r]) ) - tf.exp(-self.tf_D*Xsum)*(tf.erf(Y/self.lr[r] - self.v_r[r]) + tf.erf(self.v_r[r]))
            G = tf.exp(tf.pow(self.v_r[r],2))/(2*self.tf_D)*G 

            K = K + 0.5*np.sqrt(np.pi)*tf.exp(self.tf_log_s[r]*2)*self.lr[r]*(H + tf.transpose(G))
        return K


    #directly compute
    #X: N by 1
    def kernel_matrix2(self, X):
        N = tf.shape(X)[0]
        d = tf.shape(X)[1]
        Xdiff = X - tf.transpose(X)
        Xsum = X + tf.transpose(X)
        K = (jitter+tf.exp(self.tf_log_s0))*tf.eye(N)
        for r in range(self.R):
            H = tf.exp(-self.tf_D*Xdiff)*(tf.erf(Xdiff/self.lr[r] - self.v_r[r]) + tf.erf(tf.transpose(X)/self.lr[r] + self.v_r[r]) ) - tf.exp(-self.tf_D*Xsum)*(tf.erf(X/self.lr[r] - self.v_r[r]) + tf.erf(self.v_r[r]))
            H = tf.exp(tf.pow(self.v_r[r],2))/(2*self.tf_D)*H 
            K = K + 0.5*np.sqrt(np.pi)*tf.exp(self.tf_log_s[r]*2)*self.lr[r]*(H + tf.transpose(H))
        return K

        


    def kernel_matrix(self, ker_func, X):
        N = tf.shape(X)[0]
        d = tf.shape(X)[1]
        Xl = tf.reshape(tf.tile(tf.expand_dims(tf.tile(X,[1,1]), 1), [1, N, 1]), [N*N,d])
        Xr = tf.reshape(tf.tile(tf.expand_dims(tf.tile(X,[1,1]), 0), [N, 1, 1]), [N*N,d])
        K = tf.reshape(tf.map_fn(lambda x: ker_func(x[0], x[1]), (Xl, Xr), dtype=tf.float32), [N, N]) 
        K = K + jitter*tf.eye(N)
        return K


    def kernel_cross(self, ker_func, X, Y):
        M = tf.shape(X)[0]
        N = tf.shape(Y)[0]
        d = tf.shape(Y)[1]
        Xl = tf.reshape(tf.tile(tf.expand_dims(tf.tile(X,[1,1]), 1), [1, N, 1]), [N*M,d])
        Xr = tf.reshape(tf.tile(tf.expand_dims(tf.tile(Y,[1,1]), 0), [M, 1, 1]), [N*M,d])
        K = tf.reshape(tf.map_fn(lambda x: ker_func(x[0], x[1]), (Xl, Xr), dtype=tf.float32), [M, N]) 
        return K

    def pred(self):
        #S = self.kernel_matrix(self.k_uu, self.tf_X) + 1.0/tf.exp(self.tf_log_tau)*tf.eye(self.N)
        #Kmn = self.kernel_cross(self.k_uu, self.tf_Xt, self.tf_X)
        S = self.kernel_matrix2(self.tf_X) + 1.0/tf.exp(self.tf_log_tau)*tf.eye(self.N)
        Kmn = self.kernel_cross2(self.tf_Xt, self.tf_X)
        Kmm = self.kernel_matrix2(self.tf_Xt)
        pred_mean = tf.matmul(Kmn, tf.matrix_solve(S, self.tf_y))
        pred_var = tf.diag_part(Kmm) - tf.reduce_sum(Kmn*tf.transpose(tf.matrix_solve(S, tf.transpose(Kmn))), 1)
        pred_var = tf.reshape(pred_var, [-1,1])
        return (pred_mean, pred_var)
 
    def test(self, X_star):
        pred_mean,pred_var = self.pred()
        #print (X_star.shape)
        #return  self.sess.run([pred_mean], {self.tf_Xt: X_star, self.tf_X: self.X, self.tf_y:self.y})
        return  self.sess.run([pred_mean, pred_var], {self.tf_Xt: X_star, self.tf_X: self.X, self.tf_y:self.y})

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, lam=0.0):
        '''
        tf_dict = {self.tf_X:self.X, self.tf_Xg:self.X, self.tf_y:self.y, self.tf_lam:0.0}
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        

        print('tau = %g'%(np.exp(self.tf_log_tau.eval(session=self.sess))))
        print('amplitude = %g'%(np.exp(self.tf_log_amplitude.eval(session=self.sess))))
        print('length-scale')
        print(np.exp(self.tf_log_lengthscale.eval(session=self.sess)))
        print('alpha')
        print(np.exp(self.tf_log_alpha.eval(session=self.sess)))
        '''

        #start with
        '''
        tf_dict = {self.tf_X:self.X, self.tf_y:self.y}
        S = self.S.eval(session=self.sess, feed_dict=tf_dict)
        print(S)
        w,v = np.linalg.eig(S)
        print(np.diag(v))
        '''
        '''
        tf_dict = {self.tf_X:self.X, self.tf_y:self.y}
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        
        print('tau = %g, D = %g'%(np.exp(self.tf_log_tau.eval(session=self.sess)), np.exp(self.tf_D.eval(session=self.sess))))
        print('length-scale')
        print(np.exp(self.tf_log_lr2.eval(session=self.sess)))
        print('S')
        print(np.exp(self.tf_log_s.eval(session=self.sess)))
        '''

        tf_dict = {self.tf_X:self.X, self.tf_y:self.y}
        for i in range(3000):
            self.sess.run(self.minimizer,feed_dict = tf_dict)
            if i%100==0:
                y_pred = model.test(X_test)
                y_mean = y_pred[0]
                y_var = y_pred[1]
                error = np.linalg.norm(y_mean - y_test, 2)/np.linalg.norm(y_test, 2)
                print('epoch %d, Error u: %e' % (i,error))                     



        
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
    test = data['data']['test'][0][0]

    #extrapolation
    Xtr = X_test[0:101,:]
    X_test = X_test[101:,:]
    ytr = y_test[0:101,:]
    y_test = y_test[101:,:]
    print(Xtr.shape)
    print(X_test.shape)
    print(ytr.shape)
    print(y_test.shape)

    model = LFM(Xtr, ytr, 1)
    model.train()
    [y_mean, y_var] = model.test(X_test)
    [y_mean_tr, y_var_tr] = model.test(Xtr)
    rmse = np.linalg.norm(y_mean - y_test, 2)
    nrmse = rmse/np.linalg.norm(y_test, 2)
    res = {'rmse':rmse, 'nrmse':nrmse, 'xtr':Xtr, 'ytr':ytr, 'pred_mean_tr':y_mean_tr, 'pred_var_tr':y_var_tr, 'xtest':X_test, 'y_test':y_test, 'pred_mean':y_mean, 'pred_var':y_var }
    print('rmse = %g, nrmse = %g' % (rmse, nrmse))                     
    scipy.io.savemat('res_LFM.mat',res)
