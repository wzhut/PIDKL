import numpy as np
from scipy.io import loadmat
import PIGP_pm_post as PIGP




if __name__ == '__main__':
    layers = [3, 20, 20, 20, 20]
    g_list = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    g_idx = 2
    gamma = g_list[g_idx]
    error = []
    for nfold in range(1, 6):
        infile = 'f' + str(nfold) + '.mat'
        data = loadmat(infile, squeeze_me=True, struct_as_record=False, mat_dtype=True)

        X_train = data['X_train']
        Y_train = data['Y_train']
        X_test = data['X_test']
        Y_test = data['Y_test']
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
        X_all = np.concatenate((X_train, X_test), axis=0)
        cfg = {
            'X_train': X_train,
            'y_train': Y_train, 
            'X_test': X_test,
            'y_test': Y_test,
            'imax': np.max(X_all, axis=0),
            'imin': np.min(X_all, axis=0),
            'batch_sz': 10,
            'input_dim': 3,
            'output_dim': 1,
            'gamma': gamma,
            'layers': layers,
            'nepoch': 10000,
            'lr': 1e-3
        }

        model = PIGP.PIGPR(cfg)
        ofile = 'g' + str(g_idx + 1) + 'f' + str(nfold) + '.txt' 
        try:
            error, ll = model.train(ofile)
        except:
            continue