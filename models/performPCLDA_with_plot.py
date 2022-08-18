import numpy as np
import os, pdb
import pandas as pd
import plotly.express as px

from config import current_config as config
from models.utils_models import reduceDim, plotScatterPlotly, computeEvalMetrics, plotAndSaveConfusionMatrix, getDisplayList, plotScatterMatplotlib

def performPCLDA_with_plot(x_train, y_train, x_test, y_test, params=None):

    # Perform PC-LDA
    _, x_test, y_test_predicted = reduceDim(x_train, y_train, x_test, 'pc-lda', params)
    
    # Generate a Pandas dataframe for the test data
    column_names = []
    for idx in range(x_test.shape[1]):
        column_names.append('comp '+str(idx+1))

    df_test = pd.DataFrame(data=x_test, columns = column_names)
    
    # Convert the labels from integers to names
    y_test_names = []
    for y in y_test:
        y_test_names.append(params['class_list'][int(y)])

    df_test['category']=y_test_names
    
    # Determine the combinations that we want to visualize
    max_comp = min(config.largest_LDA_component_for_visualization,x_test.shape[1])
    
    desired_component_number_list = getDisplayList(max_comp,
                                                   config.lda_display_list)
    
       
    # Visualize and save the figures/videos
    for idx_display_set, desired_component_number in enumerate(desired_component_number_list):
        
        temp_outfile_path = os.path.join(params['results_path'],
                                         'set'+str(idx_display_set+1))
        
        
        desired_component_number_mod = []
        for curr_comp in desired_component_number:
            if curr_comp <= max_comp:
                desired_component_number_mod.append(curr_comp)
                
        col_name_list = []
        for idx in range(len(desired_component_number_mod)):
            col_name_list.append('comp '+str(desired_component_number_mod[idx]))
        
        plotScatterPlotly(df_test, 
                          col_name_list=col_name_list,
                          method_name='pc-lda',
                          results_outfile_path_no_extension=temp_outfile_path,
                          save_html=config.save_html, 
                          save_video=config.save_video,
                          params=None)
        
        plotScatterMatplotlib(df_test, 
                          col_name_list=col_name_list,
                          method_name='pc-lda',
                          results_outfile_path_no_extension=temp_outfile_path,
                          save_video=config.save_video,
                          params=params)

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

