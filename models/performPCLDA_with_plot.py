import numpy as np
import os, pdb
import pandas as pd
import plotly.express as px

from config import current_config as config
from models.utils_models import reduceDim, plotScatter3D, computeClassifAccuracy, plotAndSaveConfusionMatrix

def performPCLDA_with_plot(x_train, y_train, x_test, y_test, params=None):

    # Perform PC-LDA
    _, x_test, y_test_predicted = reduceDim(x_train, y_train, x_test, 'pc-lda', params)
    
    # Generate a Pandas dataframe for the test data
    column_names = []
    for idx in range(x_test.shape[1]):
        column_names.append('component '+str(idx+1))

    df_test = pd.DataFrame(data=x_test, columns = column_names)
    
    # Convert the labels from integers to names
    y_test_names = []
    for y in y_test:
        y_test_names.append(params['class_list'][int(y)])

    df_test['category']=y_test_names
    
    
    # Determine the combinations that we want to visualize
    max_comp = min(config.largest_LDA_component_for_visualization,x_test.shape[1])
    if len(config.lda_display_list) == 0:
        
        desired_component_number_list = []
        for idx1 in range(max_comp):
            for idx2 in np.arange(idx1+1,max_comp):
                for idx3 in np.arange(idx2+1,max_comp):
                    desired_component_number_list.append([idx1+1,idx2+1,idx3+1])
    else:
        desired_component_number_list = config.lda_display_list
       
    # Visualize and save the figures/videos
    for idx_display_set, desired_component_number in enumerate(desired_component_number_list):
        
        temp_outfile_path = os.path.join(params['results_path'],'set'+str(idx_display_set+1))
        
        
        plotScatter3D(df_test, 
                      col_name_for_x='component '+str(desired_component_number[0]), 
                      col_name_for_y='component '+str(desired_component_number[1]), 
                      col_name_for_z='component '+str(desired_component_number[2]),
                      method_name='pc-lda',
                      results_outfile_path_no_extension=temp_outfile_path,
                      save_html=config.save_html, 
                      save_video=config.save_video,
                      params=None)
        
    ## Compute the accuracy
    acc = computeClassifAccuracy(y_test, y_test_predicted)
    
    
    plotAndSaveConfusionMatrix(y_test, 
                               y_test_predicted, 
                               params['class_list'],
                               params['results_path'])
    
    return acc
    
