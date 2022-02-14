from importlib.metadata import requires
import numpy as np
import torch 
from tqdm import tqdm
from scipy.io import savemat, loadmat
import util

class LFM:
    def __init__(self, cfg):
        self.jitter = cfg['jitter']
        self.lr = cfg['lr']
        self.nepoch = cfg['nepoch']
        self.test_every = cfg['test_every']
        self.rank = cfg['rank']
        self.device = torch.device(cfg['device'])
        self.t = cfg['time']

        # data
        # data
        self.train_X = torch.tensor(cfg['train_X'], dtype=torch.float64, device=self.device)
        self.train_y = torch.tensor(cfg['train_y'], dtype=torch.float64, device=self.device)
        self.test_X = torch.tensor(cfg['test_X'], dtype=torch.float64, device=self.device)
        self.test_y = torch.tensor(cfg['test_y'], dtype=torch.float64, device=self.device)

        self.input_dim = self.train_X.shape[1]
        self.output_dim = self.train_y.shape[1]

        self.log_v = torch.tensor(np.zeros(self.rank), device=self.device, dtype=torch.float64, requires_grad=True) #self.register_param(t.tensor(np.zeros(self.rank)))
        self.log_v_0 = torch.tensor(np.zeros(self.rank), device=self.device, dtype=torch.float64, requires_grad=True) #self.register_param(t.tensor(np.zeros(self.rank)))

        self.log_a = torch.tensor([0.], device=self.device, dtype=torch.float64, requires_grad=True)# self.register_param(t.tensor([0.]))
        self.log_alpha = torch.tensor(0.1 * np.random.randn(self.rank), device=self.device, dtype=torch.float64, requires_grad=True) #self.register_param(t.tensor(0.1 * np.random.randn(self.rank)))
        self.log_tau = torch.tensor([20.], dtype=torch.float64, device=self.device, requires_grad=True)

        self.params = [self.log_tau, self.log_v, self.log_v_0, self.log_a, self.log_alpha]
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

    def train(self):
        best_rmse = 100
        for epoch in tqdm(range(self.nepoch), ascii=True, desc='Total'):
            self.optimizer.zero_grad()
            nELBO = self.get_nELBO()
            nELBO.backward()
            self.optimizer.step()
                
            if epoch % self.test_every == 0:
                with torch.no_grad():
                    rmse, nrmse, ll = self.test(self.test_X, self.test_y)
                    print('Epoch: %d RMSE: %f, nRMSE: %f, Test LL: %f' % (epoch+1, rmse, nrmse, ll))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        tr_pred_mean, tr_pred_std = self.pred(self.train_X)
                        te_pred_mean, te_pred_std = self.pred(self.test_X)
                        savemat('LFM_res.mat', {'tr_pred_mean': tr_pred_mean.tolist(), 
                                                  'tr_pred_std': tr_pred_std.tolist(),
                                                  'te_pred_mean': te_pred_mean.tolist(),
                                                  'te_pred_std': te_pred_std.tolist(),
                                                  'rmse': rmse.item(),
                                                  'nrmse': nrmse.item(),
                                                  'll': ll.item()})

    def get_nELBO(self):
        log_lengthscale = torch.log(8 * torch.exp(self.log_a) * self.t + torch.exp(self.log_v))
        log_amp = 2 * self.log_alpha + self.log_v_0 + 0.5 * self.input_dim * (self.log_v - log_lengthscale)

        S = self.kernel_matrix(log_amp,log_lengthscale)
        S = S + 1.0 / torch.exp(self.log_tau) * torch.eye(self.train_X.shape[0], device=S.device, dtype=S.dtype)
        # L = 0.5 * t.logdet(S) + 0.5 * tf.matmul(tf.transpose(self.tf_y), tf.matrix_solve(S, self.tf_y))[0,0]
        L = 0.5 * torch.logdet(S) + 0.5 * self.train_y.T @ torch.linalg.solve(S, self.train_y) 
        return L
    
    def kernel_matrix(self, log_amp, log_lengthscale):
        #rbf kernel
        norm1 = (self.train_X ** 2).sum(dim=1).view((-1, 1))
        norm2 = (self.train_X ** 2).sum(dim=1).view((1, -1))
        K = norm1 - 2.0 * self.train_X @ self.train_X.T + norm2
        retK = torch.zeros_like(K)
        for r in range(self.rank):
            retK = retK + torch.exp(log_amp[r]) * torch.exp(-1.0 / torch.exp(log_lengthscale[r])*K)
        retK = retK + self.jitter * torch.eye(self.train_X.shape[0], device=retK.device, dtype=retK.dtype)
        return retK

    def kernel_cross(self, X1, X2, log_amp, log_lengthscale):
        norm1 = (X1 ** 2).sum(dim=1).view((-1, 1))
        norm2 = (X2 ** 2).sum(dim=1).view((1, -1))
        K = norm1 - 2.0 * X1 @ X2.T + norm2
        retK = torch.zeros_like(K)
        for r in range(self.rank):
            retK = retK + torch.exp(log_amp[r]) * torch.exp(-1.0 / torch.exp(log_lengthscale[r]) * K)
        return retK

    def test(self, X, y):
        y_mean, y_std = self.pred(X)
        y_var = y_std ** 2 + torch.exp(-self.log_tau)
        rmse = torch.sqrt(((y - y_mean)**2).mean())
        nrmse = torch.norm(y - y_mean) / torch.norm(y)
        ll = -0.5 * torch.log(2 * np.pi * y_var) - 0.5 * (y - y_mean)**2 / y_var
        ll = ll.mean()
        return rmse, nrmse, ll
    
    def pred(self, X):
        log_lengthscale = torch.log(8 * torch.exp(self.log_a) * self.t + torch.exp(self.log_v))
        log_amp = 2 * self.log_alpha + self.log_v_0 + 0.5 * self.input_dim * (self.log_v - log_lengthscale)

        S = self.kernel_matrix(log_amp,log_lengthscale)
        S = S + 1.0 / torch.exp(self.log_tau) * torch.eye(self.train_X.shape[0], device=S.device, dtype=S.dtype)
        
        Kmn = self.kernel_cross(X, self.train_X, log_amp,log_lengthscale)
        # pred_mean = tf.matmul(Kmn, tf.matrix_solve(S, self.tf_y))
        pred_mean = Kmn @ torch.linalg.solve(S, self.train_y)
        # pred_var = tf.reduce_sum(tf.exp(self.tf_log_amp)) + jitter - tf.reduce_sum(Kmn*tf.transpose(tf.matrix_solve(S, tf.transpose(Kmn))), 1)
        pred_std = torch.exp(log_amp).sum() + self.jitter - (Kmn * torch.linalg.solve(S, Kmn.T).T).sum(1)
        pred_std = torch.sqrt(pred_std).view((-1, 1))
        return (pred_mean, pred_std)

if __name__ == '__main__':
    data = loadmat('./diffuBC_1d_v1.mat')
    train_X = data['xtr_t15']
    train_y = data['ytr_t15'].reshape((-1, 1))

    test_X = data['X']
    test_y = data['Y'].reshape((-1, 1))

    cfg = {
        'device': 'cpu',
        'jitter': 1e-5,
        'lr': 5e-2,
        'nepoch': 5000,
        'test_every': 1,
        'train_X': train_X[:, 0].reshape((-1, 1)),
        'train_y': train_y,
        'test_X': test_X[:, 0].reshape((-1, 1)),
        'test_y': test_y,
        'rank': 1,
        'time': 0.5
    }

    model = LFM(cfg)
    model.train()