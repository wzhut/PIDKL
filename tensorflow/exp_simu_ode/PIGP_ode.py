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
import time
import scipy


np.random.seed(0)
tf.set_random_seed(0)

jitter = 1e-5
jitter_SE = 1e-3

class PIGPR:

    #X_f are a list, each elemeent is a set of inputs
    def __init__(self, X_u, u, X_f, layers, lb, ub):
        self.lb = lb
        self.ub = ub
    
        self.x_u = X_u
        self.u = u
        self.layers = layers
        self.N = X_u.shape[0]
        self.M = X_f[0].shape[0]

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        #input
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        #output
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        #reg points
        self.x_f_tf = [tf.constant(X_f[i], dtype=tf.float32) for i in range(len(X_f))]


        #model parameters
        self.tf_log_lengthscale = tf.Variable(0.0, dtype=tf.float32)
        self.tf_log_amp = tf.Variable(0.0, dtype=tf.float32)
        self.tf_log_s0 = tf.Variable(0.0, dtype=tf.float32)
        #self.tf_log_amp = tf.constant(0.0, dtype=tf.float32)
        self.tf_log_tau = tf.Variable(0.0, dtype=tf.float32)
        self.tf_log_lengthscale_SE = tf.Variable(0.0, dtype=tf.float32)
        self.tf_log_D = tf.Variable(0.0, dtype=tf.float32)
        self.tf_B = tf.Variable(0.0, dtype=tf.float32)
        self.u_input = self.neural_net(self.x_u_tf, self.weights, self.biases) 
        Knn = self.kernel_matrix(self.u_input)
        self.S = Knn + 1.0/tf.exp(self.tf_log_tau)*tf.eye(self.N)
        L = 0.5*self.N*np.log(2.0*np.pi) + 0.5*tf.linalg.logdet(self.S) + 0.5*tf.matmul(tf.transpose(self.u_tf), tf.matrix_solve(self.S, self.u_tf))[0,0]
        L_reg = 0.0
        for i in range(len(self.x_f_tf)):
            Knn_g = self.kernel_matrix_SE(self.x_f_tf[i])
            f_pred = self.pde_mean(self.x_f_tf[i])
            L_reg = L_reg + 0.5*self.M*np.log(2.0*np.pi) +  0.5*tf.linalg.logdet(Knn_g) \
                  + 0.5*tf.matmul(tf.transpose(f_pred), tf.matrix_solve(Knn_g, f_pred))[0,0]

        #lam = 0.1
        lam = 0.15
        self.loss = L + lam*L_reg/len(self.x_f_tf)
        #self.loss = L 
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))


        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        '''
        self.sess = tf.Session(config=tf.ConfigProto(
            # log_device_placement=True,
            # device_count={'GPU': 1},
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
        )) 
        self.sess.run(tf.global_variables_initializer())
        '''

        init = tf.global_variables_initializer()
        self.sess.run(init)

    #ODE
    def pde_mean(self, t):
        u = self.pred_mean(t)
        u_t = tf.gradients(u, t)[0]
        f = u_t + tf.exp(self.tf_log_D)*u - self.tf_B
        return f

    
    def pred_mean(self, t):
        inputs = self.neural_net(t, self.weights, self.biases)
        #Kmn = self.spectral_mixture_kernels(inputs, self.u_input)
        Kmn = self.kernel_cross(inputs, self.u_input)
        post_mean = tf.matmul(Kmn, tf.matrix_solve(self.S, self.u_tf))
        return post_mean

    def test(self, X_star):
        X_star_tf = tf.placeholder(tf.float32)
        input_star = self.neural_net(X_star_tf, self.weights, self.biases)
        Kmn = self.kernel_cross(input_star, self.u_input)
        post_mean = tf.matmul(Kmn, tf.matrix_solve(self.S, self.u_tf))
        pred_var = tf.exp(self.tf_log_amp) + jitter - tf.reduce_sum(Kmn*tf.transpose(tf.matrix_solve(self.S, tf.transpose(Kmn))), 1)
        pred_var = tf.reshape(pred_var,[-1,1])
        print('tau = %g, D = %g,B = %g,  amplitute = %g, length-scale = %g, length-scale-LF = %g, s0 = %g'%(np.exp(self.tf_log_tau.eval(session=self.sess)),np.exp(self.tf_log_D.eval(session=self.sess)), self.tf_B.eval(session=self.sess), np.exp(self.tf_log_amp.eval(session=self.sess)),  np.exp(self.tf_log_lengthscale.eval(session=self.sess)), np.exp(self.tf_log_lengthscale_SE.eval(session=self.sess)), np.exp(self.tf_log_s0.eval(session=self.sess))))

        tf_dict = {self.x_u_tf:self.x_u, self.u_tf: self.u, X_star_tf:X_star}
        return self.sess.run([post_mean,pred_var], tf_dict)

    #SE
    def kernel_matrix_SE(self, tf_X):
        col_norm2 = tf.reduce_sum(tf_X*tf_X, 1)
        col_norm2 = tf.reshape(col_norm2, [-1,1])
        K = col_norm2 - 2.0*tf.matmul(tf_X, tf.transpose(tf_X)) + tf.transpose(col_norm2)
        K = tf.exp(-1.0/tf.exp(self.tf_log_lengthscale_SE)*K)
        K = K + jitter_SE*tf.eye(tf.shape(tf_X)[0])
        return K



    def kernel_matrix(self, tf_X):
        #rbf kernel
        col_norm2 = tf.reduce_sum(tf_X*tf_X, 1)
        col_norm2 = tf.reshape(col_norm2, [-1,1])
        K = col_norm2 - 2.0*tf.matmul(tf_X, tf.transpose(tf_X)) + tf.transpose(col_norm2)
        K = tf.exp(self.tf_log_amp)*tf.exp(-1.0/tf.exp(self.tf_log_lengthscale)*K)
        K = K + (jitter+tf.exp(self.tf_log_s0))*tf.eye(tf.shape(K)[0])
        #K = K + (jitter)*tf.eye(tf.shape(K)[0])
        return K

    def kernel_cross(self, tf_Xt, tf_X):
        col_norm1 = tf.reshape(tf.reduce_sum(tf_Xt*tf_Xt, 1), [-1, 1])
        col_norm2 = tf.reshape(tf.reduce_sum(tf_X*tf_X, 1), [-1, 1])
        K = col_norm1 - 2.0*tf.matmul(tf_Xt, tf.transpose(tf_X)) + tf.transpose(col_norm2)
        K = tf.exp(self.tf_log_amp)*tf.exp(-1.0/tf.exp(self.tf_log_lengthscale)*K)
        return K

    def kernel_diag(self, tf_Xt):
        return tf.exp(self.tf_log_amp)*tf.ones(tf.shape(tf_Xt)[0]) + jitter

    def spectral_mixture_kernel_matrix(self, tf_X):
        K = self.spectral_mixture_kernels(tf_X, tf_X)
        return K + jitter*tf.eye(tf.shape(K)[0])

    def spectral_mixture_kernels(self,tf_Xt, tf_X):

        return_tensor = tf.zeros(shape=[tf.shape(tf_Xt)[1],tf.shape(tf_X)[1]])

        print('tf_Xt.shape',tf_Xt.shape)
        print('tf_X.shape',tf_X.shape)
        tf_Xt = tf.expand_dims(tf_Xt,axis=0)
        tf_X = tf.expand_dims(tf_X,axis=0)
        tf_Xt = tf.tile(tf_Xt,[tf.shape(tf_X)[1],1,1])
        tf_X = tf.tile(tf_X,[tf.shape(tf_Xt)[1],1,1])

        print('tf_Xt.shape1',tf_Xt.shape)
        print('tf_X.shape1',tf_X.shape)

        diff_mat = tf.transpose(tf_Xt,perm=[1,0,2]) - (tf_X)
        diff_mat = tf.expand_dims(diff_mat,axis = 2)

        for i in range(spectral_mixture_kernels_number):

            mixture_weight = self.spectral_mixture_kernels_parameters[i]['mixture_weight']
            bandwidths = self.spectral_mixture_kernels_parameters[i]['bandwidths']
            frequency = self.spectral_mixture_kernels_parameters[i]['frequency']

            term_1 = mixture_weight * tf.sqrt(tf.norm(bandwidths,ord=2)) / tf.pow(2*tf.constant(np.pi,dtype=tf.float32),tf.cast(tf.shape(tf_Xt)[2]/2, dtype=tf.float32))

            bandwidths = tf.expand_dims(bandwidths,axis = 0)
            bandwidths = tf.expand_dims(bandwidths,axis = 0)
            bandwidths = tf.tile(bandwidths,[tf.shape(tf_Xt)[1],tf.shape(tf_X)[1],1,1])

            term_2 = tf.matmul(diff_mat,tf.sqrt(bandwidths))
            term_2 = tf.matmul(term_2,tf.transpose(diff_mat,perm=[0,1,3,2]))
            term_2 = tf.exp(term_2/(-2))
            term_2 = tf.reshape(term_2,shape = [tf.shape(tf_Xt)[1],tf.shape(tf_X)[1]])

            frequency = tf.expand_dims(frequency,axis=0)
            frequency = tf.expand_dims(frequency,axis=0)
            frequency = tf.tile(frequency,[tf.shape(tf_Xt)[1],tf.shape(tf_X)[1],1,1])
            tp = tf.matmul((2*tf.constant(np.pi) * frequency),tf.transpose(diff_mat,[0,1,3,2]))
            term_3 = tf.cos(tp)
            term_3 = tf.reshape(term_3,shape = [tf.shape(tf_Xt)[1],tf.shape(tf_X)[1]])

            return_tensor += tf.multiply(tf.multiply(term_1,term_2),term_3)

        return return_tensor

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            #W = self.msra_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)        
        return weights, biases

    def initialize_NN_with_PINN(self, layers):
        w = np.load('w.npy')
        b = np.load('b.npy')
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0, num_layers-1):
            tf_W = tf.Variable(w[l],dtype=tf.float32)
            tf_b = tf.Variable(b[l],dtype=tf.float32)
            weights.append(tf_W)
            biases.append(tf_b)        
        return weights, biases

                
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev))

    def msra_init(self,size):
        in_dim = size[0]
        out_dim = size[1]    
        msra_stddev = np.sqrt(2.0/(in_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=msra_stddev))


    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            #H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def callback(self, loss):
        #global latest_loss
        #latest_loss = loss
        print('Loss:', loss)
        return

    def train(self):
        tf_dict = {self.x_u_tf: self.x_u, self.u_tf: self.u}
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)

def to_vec(x):
    return x.reshape([x.size,1])

if __name__ == '__main__':
    #layers = [3, 20, 20, 20, 20]
    layers = [1, 20, 20, 20, 20]
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

    D = 5
    X_f = [np.random.uniform(size=(100,1)) for i in range(D)]
    model = PIGPR(Xtr, ytr, X_f, layers, 0, 1)
    model.train()
    [y_mean, y_var] = model.test(X_test)
    [y_mean_tr, y_var_tr] = model.test(Xtr)
    rmse = np.linalg.norm(y_mean - y_test, 2)
    nrmse = rmse/np.linalg.norm(y_test, 2)
    rmse = np.sqrt(np.mean((y_mean - y_test)**2))
    res = {'rmse':rmse, 'nrmse':nrmse, 'xtr':Xtr, 'ytr':ytr, 'pred_mean_tr':y_mean_tr, 'pred_var_tr':y_var_tr, 'xtest':X_test, 'y_test':y_test, 'pred_mean':y_mean, 'pred_var':y_var }
    print('rmse = %g, nrmse = %g' % (rmse, nrmse))                     
    scipy.io.savemat('res_PIGP.mat',res)
