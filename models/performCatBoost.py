import numpy as np
from catboost import CatBoostClassifier
from config import current_config as config
from models.utils_models import reduceDim, computeClassifAccuracy, plotAndSaveConfusionMatrix
    
    
def performCatBoost(x_train, y_train, x_test, y_test, preproc_type=None, params=None):
    
    ## Reduce the dimension of the data if specified
    if preproc_type == 'pca':
        x_train, x_test,_ = reduceDim(x_train, y_train, x_test, preproc_type, params)
    elif preproc_type == 'pc-lda':
        x_train, x_test,_ = reduceDim(x_train, y_train, x_test, preproc_type, params)

    # Train the model
    clf = CatBoostClassifier(iterations=500,
                             depth=4,
                             loss_function='Logloss',
                             verbose=True)

    # Train the model
    clf.fit(x_train,y_train)

    # Test the model
    y_test_predicted = clf.predict(x_test)
        
        
    ## Compute the accuracy
    acc = computeClassifAccuracy(y_test, y_test_predicted)
    
    
    plotAndSaveConfusionMatrix(y_test, 
                               y_test_predicted, 
                               params['class_list'],
                               params['results_path'])
    
    return acc

    