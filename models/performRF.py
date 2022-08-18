from sklearn.ensemble import RandomForestClassifier
from config import current_config as config
from models.utils_models import reduceDim, computeEvalMetrics, plotAndSaveConfusionMatrix
    
    
def performRF(x_train, y_train, x_test, y_test, preproc_type=None, params=None):
    
    ## Reduce the dimension of the data if specified
    if preproc_type == 'pca':
        x_train, x_test,_ = reduceDim(x_train, y_train, x_test, preproc_type, params)
    elif preproc_type == 'pc-lda':
        x_train, x_test,_ = reduceDim(x_train, y_train, x_test, preproc_type, params)

    ## Train the model
    if config.perform_hparam_tuning:
        from tune_sklearn import TuneGridSearchCV
        
        hparams = {
            # 'n_estimators':[10,100],
            'max_depth':[3,4,5],
            'max_features':['sqrt','log2']
        }
        
        tune_search = TuneGridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=hparams,
            early_stopping=True,
            max_iters=100
        )
        
        # Train the model
        tune_search.fit(x_train,y_train)
        
        # Test the model
        y_test_predicted = tune_search.predict(x_test)
            
    else:
        clf = RandomForestClassifier()
        
        # Train the model
        clf.fit(x_train,y_train)
        
        # Test the model
        y_test_predicted = clf.predict(x_test)

    ## Compute evaluation metrics
    eval_metrics_dict = computeEvalMetrics(y_test,
                                           y_test_predicted,
                                           params['class_list'],
                                           params['results_path'])

    plotAndSaveConfusionMatrix(y_test,
                               y_test_predicted,
                               params['class_list'],
                               params['results_path'])

    return eval_metrics_dict

    