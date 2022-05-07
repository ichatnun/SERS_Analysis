import numpy as np
import os, pdb
import pandas as pd
import plotly.express as px

from config import current_config as config
from models.utils_models import reduceDim, plotScatter3D

def performPCA_with_plot(x_train, y_train, x_test, y_test, params=None):
    
    # Perform PCA
    _, x_test, explained_var = reduceDim(x_train, y_train, x_test, 'pca', params)
    
    # Generate a Pandas dataframe for the test data
    column_names = []
    for idx in range(x_test.shape[1]):
        column_names.append('PC'+str(idx+1))

    df_test = pd.DataFrame(data=x_test, columns = column_names)
    
    # Convert the labels from integers to names
    y_test_names = []
    for y in y_test:
        y_test_names.append(params['class_list'][int(y)])

    df_test['category']=y_test_names
    
    # Determine the PC combinations that we want to visualize
    if len(config.pca_display_list) == 0:

        max_PC = config.largest_PC_for_visualization
        
        desired_component_number_list = []
        for idx1 in range(max_PC):
            for idx2 in np.arange(idx1+1,max_PC):
                for idx3 in np.arange(idx2+1,max_PC):
                    desired_component_number_list.append([idx1+1,idx2+1,idx3+1])
    else:
        desired_component_number_list = config.pca_display_list
       
    # Visualize and save the figures/videos
    for idx_display_set, desired_component_number in enumerate(desired_component_number_list):
        
        temp_outfile_path = os.path.join(params['results_path'],'set'+str(idx_display_set+1))
        
        temp_explained_var = [explained_var[desired_component_number[0]-1],
                              explained_var[desired_component_number[1]-1],
                              explained_var[desired_component_number[2]-1]]
        
        plotScatter3D(df_test, 
                      col_name_for_x='PC'+str(desired_component_number[0]), 
                      col_name_for_y='PC'+str(desired_component_number[1]), 
                      col_name_for_z='PC'+str(desired_component_number[2]),
                      method_name='pca',
                      results_outfile_path_no_extension=temp_outfile_path,
                      save_html=config.save_html, 
                      save_video=config.save_video,
                      params={'explained_var':temp_explained_var})
    
