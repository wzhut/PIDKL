import numpy as np
from scipy.io import loadmat

if __name__ == '__main__':
    # rrmse = np.array([0.5708831411,	0.4959948183,	0.6345629441,	0.5614590373,	0.5995008661])
    # rrmse = np.array([0.6750250843,	0.5206723719,	0.9023275173,	0.5693146381,	0.6411589243]) 
    
    rrmse = np.array([0.5941389129,	0.5771925206,	0.5427324121,	0.6051284805,	0.5203567553]) 
    rmses = []
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
        n_test = X_test.shape[0]
        rmse = (rrmse[nfold-1] * np.linalg.norm(Y_test, 2)) / np.sqrt(n_test)
        rmses.append(rmse)
    print(rmses)
