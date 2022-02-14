import numpy as np
from scipy.io import loadmat
import GPR_deep as GPR




if __name__ == '__main__':
    layers = [4, 20, 20, 20, 20]
    for nfold in range(1, 6):
        infile = 'f' + str(nfold) + '.mat'
        data = loadmat(infile, squeeze_me=True, struct_as_record=False, mat_dtype=True)

        X_train = data['X_train']
        X_test = data['X_test']
        Y_train = data['Y_train']
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
        
        cfg = {
            'X_train': X_train,
            'Y_train': Y_train, 
            'X_test': X_test,
            'Y_test': Y_test,
            'input_dim': 5,
            'output_dim': 1,
            'batch_sz': 10,
            'layers': layers,
            'nepoch': 10000,
            'lr': 1e-3
        }

        model = GPR.GPR(cfg)
        ofile = 'gpr_deep_f' + str(nfold) + '.txt' 
        try:
            error, ll = model.train(ofile)
        except:
            continue
        
