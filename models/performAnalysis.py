from models.performLGBM import performLGBM
from models.performCatBoost import performCatBoost
from models.performRF import performRF
from models.performSVM import performSVM
from models.performPCA_with_plot import performPCA_with_plot
from models.performPCLDA_with_plot import performPCLDA_with_plot


def performAnalysis(method_name, params):
    
    x_train = params['x_train']
    y_train = params['y_train']
    x_test = params['x_test']
    y_test = params['y_test']
    
    if method_name == 'pca':
        performPCA_with_plot(x_train, y_train, x_test, y_test, params=params)
        
    if method_name == 'pc-lda':
        return performPCLDA_with_plot(x_train, y_train, x_test, y_test, params=params)
        
    if method_name == 'rf':
        return performRF(x_train, y_train, x_test, y_test, 
                         preproc_type=None, 
                         params=params)
        
    if method_name == 'pc-rf':
        return performRF(x_train, y_train, x_test, y_test, 
                         preproc_type='pca', 
                         params=params)
        
    if method_name == 'lda-rf':
        return performRF(x_train, y_train, x_test, y_test, 
                         preproc_type='pc-lda', 
                         params=params)

    if method_name == 'svm':
        return performSVM(x_train, y_train, x_test, y_test, 
                         preproc_type=None, 
                         params=params)

    if method_name == 'pc-svm':
        return performSVM(x_train, y_train, x_test, y_test, 
                         preproc_type='pca', 
                         params=params)
    
    if method_name == 'lda-svm':
        return performSVM(x_train, y_train, x_test, y_test, 
                         preproc_type='pc-lda', 
                         params=params)
    
    if method_name == 'catboost':
        return performCatBoost(x_train, y_train, x_test, y_test, 
                         preproc_type=None, 
                         params=params)

    if method_name == 'pc-catboost':
        return performCatBoost(x_train, y_train, x_test, y_test, 
                         preproc_type='pca', 
                         params=params)
    
    if method_name == 'lda-catboost':
        return performCatBoost(x_train, y_train, x_test, y_test, 
                         preproc_type='pc-lda', 
                         params=params)    
    
    if method_name == 'lgbm':
        return performLGBM(x_train, y_train, x_test, y_test, 
                         preproc_type=None, 
                         params=params)

    if method_name == 'pc-lgbm':
        return performLGBM(x_train, y_train, x_test, y_test, 
                         preproc_type='pca', 
                         params=params)
    
    if method_name == 'lda-lgbm':
        return performLGBM(x_train, y_train, x_test, y_test, 
                         preproc_type='pc-lda', 
                         params=params)    
    