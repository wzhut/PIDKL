from importlib.metadata import requires
from operator import itemgetter
import torch
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat, savemat
import util

np.random.seed(0)
torch.random.manual_seed(0)

class PIDKL:
    def __init__(self, cfg):
        
        # self.lb = cfg['lb']
        # self.ub = cfg['ub']
        self.layers = cfg['layers']
        # self.batch_sz = cfg['batch_sz']
        self.lr = cfg['lr']
        self.test_every = cfg['test_every']
        self.device = torch.device(cfg['device'])
        self.jitter = cfg['jitter']
        self.epoch_nb = cfg['nepoch']
        self.gamma = cfg['gamma']

        # data
        self.train_X = torch.tensor(cfg['train_X'], dtype=torch.float64, device=self.device)
        self.train_y = torch.tensor(cfg['train_y'], dtype=torch.float64, device=self.device)
        self.test_X = torch.tensor(cfg['test_X'], dtype=torch.float64, device=self.device)
        self.test_y = torch.tensor(cfg['test_y'], dtype=torch.float64, device=self.device)
        # self.co_X = torch.tensor(cfg['co_X'], dtype=torch.float64, device=self.device, requires_grad=True)
        self.co_X = None

        self.log_ls_f = torch.tensor([0.], dtype=torch.float64, device=self.device, requires_grad=True)
        self.log_ls_g = torch.tensor([0.], dtype=torch.float64, device=self.device, requires_grad=True)
        self.log_tau = torch.tensor([20.], dtype=torch.float64, device=self.device, requires_grad=True)
        self.log_D = torch.tensor([0.], dtype=torch.float64, device=self.device, requires_grad=True)

        self.nn = util.NN(self.layers).double().to(self.device)
        self.kernel_rbf = util.KernelRBF(self.jitter)

        self.params = [self.log_ls_f, self.log_ls_g, self.log_tau, self.log_D] + list(self.nn.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
        # self.optimizer = torch.optim.LBFGS(self.params, max_iter=10, lr=0.2)
        

    def train(self):
        ll_list = []
        error_list = []
        best_rmse = 1000
        for epoch in tqdm(range(self.epoch_nb), ascii=True, desc='Total'):
            def closure():
                self.optimizer.zero_grad()
                nELBO = self.get_nELBO()
                nELBO.backward()
                return nELBO
            self.optimizer.step(closure)
                
            if (epoch + 1) % self.test_every == 0:
                with torch.no_grad():
                    nrmse, rmse, ll = self.test(self.test_X, self.test_y)
                    print('ls_f: %f, ls_g: %f, tau: %f' % (torch.exp(self.log_ls_f), torch.exp(self.log_ls_g), torch.exp(self.log_tau)))
                    print('Epoch: %d RMSE: %f, nRMSE: %f LL: %f' % (epoch+1, rmse, nrmse, ll))
                    # print('Error: %e, Test LL: %e' % (error, ll)) 
                    error_list.append(nrmse)
                    ll_list.append(ll)
                    if rmse < best_rmse:
                        best_rmse = rmse
                        tr_pred_mean, tr_pred_std = self.pred(self.train_X)
                        te_pred_mean, te_pred_std = self.pred(self.test_X)
                        savemat('PIDKL_res.mat', {'tr_pred_mean': tr_pred_mean.tolist(), 
                                                  'tr_pred_std': tr_pred_std.tolist(),
                                                  'te_pred_mean': te_pred_mean.tolist(),
                                                  'te_pred_std': te_pred_std.tolist(),
                                                  'rmse': rmse.item(),
                                                  'nrmse': nrmse.item(),
                                                  'll': ll.item()})

        return ll_list
       
    def get_nELBO(self):
        # self.u_input = self.neural_net(self.x_u_tf, self.weights, self.biases) 
        N = self.train_X.shape[0]
        X = self.nn(self.train_X)
        Knn = self.kernel_rbf.matrix(X, torch.exp(self.log_ls_f))
        S = Knn + 1.0 / torch.exp(self.log_tau) * torch.eye(N, device=X.device, dtype=X.dtype)
        L = 0.5 * N * np.log(2.0 * np.pi) + 0.5 * torch.logdet(S) + 0.5 * self.train_y.T @ torch.linalg.solve(S, self.train_y)
        
        # self.co_X = torch.tensor(np.random.rand(10).reshape((-1, 1)), dtype=torch.float64, device=self.device, requires_grad=True)
        self.co_X = torch.tensor(np.random.rand(10, 2) * np.array([2, 1]), dtype=torch.float64, device=self.device, requires_grad=True)
        N_g = self.co_X.shape[0]
        Knn_g = self.kernel_rbf.matrix(self.co_X, torch.exp(self.log_ls_g))
        f_pred = self.pde(self.co_X)
        L_reg = 0.5 * N_g * np.log(2.0*np.pi) +  0.5 * torch.logdet(Knn_g) + 0.5 * f_pred.T @ torch.linalg.solve(Knn_g, f_pred)

        loss = L + self.gamma * L_reg
        return loss

    def test(self, X, y):
        y_mean, y_std = self.pred(X)
        y_var = y_std ** 2 + torch.exp(-self.log_tau)
        rmse = torch.sqrt(torch.mean((y - y_mean)**2))
        nrmse = torch.norm(y - y_mean) / torch.norm(y)
        ll = -0.5 * torch.log(2 * np.pi * y_var) - 0.5 * (y - y_mean)**2 / y_var
        ll = ll.mean()
        return rmse, nrmse, ll

    def pde(self, X_g):
        #u = self.pred_mean(t)
        u_mean,u_std = self.pred(X_g)
        eta = torch.empty_like(u_mean).normal_()
        u = u_mean + u_std * eta
        du = torch.autograd.grad(u.sum(), X_g, create_graph=True)[0]
        dudt = du[:, 1]
        d2udx2 = torch.autograd.grad(du[:, 0].sum(), X_g, create_graph=True)[0][:, 0]
        f = dudt - torch.exp(self.log_D) * d2udx2
        #f = u_t + u - 1
        return f

    def pred(self, test_X):
        ls_f = torch.exp(self.log_ls_f)

        train_X = self.nn(self.train_X)
        test_X = self.nn(test_X)

        Kmn = self.kernel_rbf.cross(test_X, train_X, ls_f)
        Knn = self.kernel_rbf.matrix(train_X, ls_f)
        N = train_X.shape[0]

        S = Knn + 1.0 / torch.exp(self.log_tau) * torch.eye(N, device=train_X.device, dtype=train_X.dtype)
        post_mean = Kmn @ torch.linalg.solve(S, self.train_y)
        pred_std = 1 + self.jitter - (Kmn * torch.linalg.solve(S, Kmn.T).T).sum(1)
        pred_std = torch.sqrt(pred_std).view((-1, 1))
        return (post_mean, pred_std)

if __name__ == '__main__':
    layers = [2, 20, 20, 20, 20]
    data = loadmat('./diffuBC_1d_v1.mat')
    train_X = data['xtr_t15']
    train_y = data['ytr_t15'].reshape((-1, 1))

    test_X = data['X']
    test_y = data['Y'].reshape((-1, 1))

    cfg = {
        'device': 'cpu',
        'layers': layers,
        'jitter': 1e-5,
        'lr': 5e-2,
        'nepoch': 5000,
        'test_every': 1,
        'train_X': train_X,
        'train_y': train_y,
        'test_X': test_X,
        'test_y': test_y,
        'gamma':  1
    }

    model = PIDKL(cfg)
    model.train()

    # [y_mean, y_var] = model.pred_mean_and_std(t.from_numpy(te_x).float().cuda(0))
    # [y_mean_tr, y_var_tr] = model.pred_mean_and_std(t.from_numpy(tr_x).float().cuda(0))

    # y_mean = y_mean.cpu().detach().numpy()
    # y_var = y_var.cpu().detach().numpy()
    # y_mean_tr = y_mean_tr.cpu().detach().numpy()
    # y_var_tr = y_var_tr.cpu().detach().numpy()

    # rmse = np.linalg.norm(y_mean - y_test, 2)
    # nrmse = rmse/np.linalg.norm(y_test, 2)

    # res = {'rmse':rmse, 'nrmse':nrmse, 'xtr':Xtr, 'ytr':ytr, 'pred_mean_tr':y_mean_tr, 'pred_var_tr':y_var_tr, 'xtest':X_test, 'y_test':y_test, 'pred_mean':y_mean, 'pred_var':y_var }
    # print('rmse = %g, nrmse = %g' % (rmse, nrmse))                     
    # scipy.io.savemat('res_PIGP.mat',res)