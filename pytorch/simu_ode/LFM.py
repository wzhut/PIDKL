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

        # data
        # data
        self.train_X = torch.tensor(cfg['train_X'], dtype=torch.float64, device=self.device)
        self.train_y = torch.tensor(cfg['train_y'], dtype=torch.float64, device=self.device)
        self.test_X = torch.tensor(cfg['test_X'], dtype=torch.float64, device=self.device)
        self.test_y = torch.tensor(cfg['test_y'], dtype=torch.float64, device=self.device)

        self.input_dim = self.train_X.shape[1]
        self.output_dim = self.train_y.shape[1]

        self.log_tau = torch.tensor([20.], dtype=torch.float64, device=self.device, requires_grad=True)
        self.log_s0 = torch.tensor([0.], dtype=torch.float64, device=self.device, requires_grad=True)
        self.log_lr2 = torch.tensor(np.zeros(self.rank), dtype=torch.float64, device=self.device, requires_grad=True)
        self.log_s = torch.tensor(np.zeros(self.rank), dtype=torch.float64, device=self.device, requires_grad=True)
        self.log_D = torch.tensor([0.], dtype=torch.float64, device=self.device, requires_grad=True) 

        self.params = [self.log_tau, self.log_s0, self.log_lr2, self.log_s, self.log_D]
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
        v_r = [None for r in range(self.rank)]
        lr = [None for r in range(self.rank)]
        for r in range(self.rank):
            lr[r] = torch.exp(0.5 * self.log_lr2[r])
            v_r[r] = 0.5 * lr[r] * torch.exp(self.log_D)

        N = self.train_X.shape[0]
        S = self.kernel_matrix2(self.train_X, lr, v_r) + 1.0 / torch.exp(self.log_tau) * torch.eye(N, device=self.train_X.device, dtype=self.train_X.dtype)
        
        negL = 0.5 * N *np.log(2.0 * np.pi) + 0.5 * torch.logdet(S) + 0.5 * self.train_y.T @ torch.linalg.solve(S, self.train_y)
        return negL

    def kernel_matrix2(self, X, lr, v_r):
        N = X.shape[0]
        d = X.shape[1]
        Xdiff = X - X.T
        Xsum = X + X.T

        K = (self.jitter + torch.exp(self.log_s0)) * torch.eye(N, device=X.device, dtype=X.dtype)
        D = torch.exp(self.log_D)

        for r in range(self.rank):
            H = torch.exp(-D * Xdiff) * (torch.erf(Xdiff / lr[r] - v_r[r]) \
                + torch.erf(X.T / lr[r] + v_r[r]) ) \
                - torch.exp(-D * Xsum) * (torch.erf(X / lr[r] - v_r[r]) \
                + torch.erf(v_r[r]))
            H = torch.exp(v_r[r] ** 2) / (2 * D)*H 
            K = K + 0.5 * np.sqrt(np.pi) * torch.exp(self.log_s[r] * 2) * lr[r] * (H + H.T)
        return K
    
    def kernel_cross2(self, X, Y, lr, v_r):
        N = X.shape[0]
        M = Y.shape[0]
        K= torch.zeros((N, M), device=X.device, dtype=X.dtype)
        D = torch.exp(self.log_D)
        for r in range(self.rank):
            Xdiff = X - Y.T
            Xsum = X + Y.T
            H = torch.exp(-D * Xdiff) * (torch.erf(Xdiff / lr[r] - v_r[r]) \
                + torch.erf(Y.T / lr[r] + v_r[r]) ) \
                - torch.exp(-D * Xsum) * (torch.erf(X /lr[r] - v_r[r]) \
                + torch.erf(v_r[r]))
            H = torch.exp(v_r[r] ** 2)/(2 * D) * H 

            Xdiff = Y - X.T
            Xsum = Y + X.T
            G = torch.exp(-D * Xdiff)*(torch.erf(Xdiff / lr[r] - v_r[r]) \
                + torch.erf(X.T / lr[r] + v_r[r])) \
                - torch.exp(-D * Xsum) * (torch.erf(Y / lr[r] - v_r[r]) \
                + torch.erf(v_r[r]))
            G = torch.exp(v_r[r] ** 2)/(2 * D) * G 

            K = K + 0.5 * np.sqrt(np.pi) * torch.exp(self.log_s[r] * 2) * lr[r] * (H + G.T)
        return K

    def test(self, X, y):
        y_mean, y_std = self.pred(X)
        y_var = y_std ** 2 + torch.exp(-self.log_tau)
        rmse = torch.sqrt(((y - y_mean)**2).mean())
        nrmse = torch.norm(y - y_mean) / torch.norm(y)
        ll = -0.5 * torch.log(2 * np.pi * y_var) - 0.5 * (y - y_mean)**2 / y_var
        ll = ll.mean()
        return rmse, nrmse, ll
    
    def pred(self, X):
        #S = self.kernel_matrix(self.k_uu, self.tf_X) + 1.0/tf.exp(self.tf_log_tau)*tf.eye(self.N)
        #Kmn = self.kernel_cross(self.k_uu, self.tf_Xt, self.tf_X)
        v_r = [None for r in range(self.rank)]
        lr = [None for r in range(self.rank)]
        for r in range(self.rank):
            lr[r] = torch.exp(0.5 * self.log_lr2[r])
            v_r[r] = 0.5 * lr[r] * torch.exp(self.log_D)

        # N = self.tr_x.shape[0]
        S = self.kernel_matrix2(self.train_X, lr, v_r) + 1.0 / torch.exp(self.log_tau) * torch.eye(self.train_X.shape[0], device=X.device, dtype=X.dtype)
        Kmn = self.kernel_cross2(X, self.train_X, lr, v_r)
        Kmm = self.kernel_matrix2(X, lr,  v_r)
        pred_mean = Kmn @ torch.linalg.solve(S, self.train_y)
        pred_std = torch.diag(Kmm).view((-1, 1)) - (Kmn * torch.linalg.solve(S, Kmn.T).T).sum(1).view((-1, 1))
        pred_std = torch.sqrt(pred_std)
        return (pred_mean, pred_std)

if __name__ == '__main__':
    #layers = [3, 20, 20, 20, 20]
    layers = [1, 20, 20, 20, 20]
    # print('layers: %s'%(str(layers)))
    data = loadmat('./ode_extrap.mat')
    train = data['train']
    test = data['test']
    print(train.shape)

    train_X = train[:, 0].reshape((-1, 1))
    train_y = train[:, 1].reshape((-1, 1))
    # train_y = train_y + np.random.randn(100).reshape((-1, 1)) * 0.05
    test_X = test[:, 0].reshape((-1, 1))
    test_y = test[:, 1].reshape((-1, 1))

    cfg = {
        'device': 'cpu',
        'jitter': 1e-5,
        'lr': 5e-2,
        'nepoch': 1000,
        'test_every': 1,
        'rank': 1,
        'train_X': train_X,
        'train_y': train_y,
        'test_X': test_X,
        'test_y': test_y,
    }

    model = LFM(cfg)
    model.train()