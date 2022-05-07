import os

# Perform hyper-parameter tuning. Otherwise, use the default parameter.
perform_hparam_tuning = False 

# The name of the folder that you want to save your results to. It would be great to make it self-explanatory.
results_folder_name = 'results'
save_html = True
save_video = False


############  Experimental details ############

# If not exist, perform the analysis on all experiments
experiment_name = 'EXO1_pos' 

# Options: 'intra', 'inter', 'by-folders'. If 'by-folders' is specified, the users must organize data as /experiment/class/train/*.txt, /experiment/class/val/*.txt, /experiment/class/test/*.txt
data_split_method = 'by-folders' 

# Options: # 'pca', 'pc-lda', 'rf', 'pc-rf', 'lda-rf', 'svm', 'pc-svm', 'lda-svm'
# ML_method_list = ['rf','pc-rf','lda-rf','pc-svm','lda-svm'] 
ML_method_list =['pca','pc-lda','rf','pc-rf','lda-rf','svm','pc-svm','lda-svm']

# Apply a baseline removal algorithm or not
REMOVE_BASELINE = False

# Default to False, unless you want to reload the data from raw files
OVERWRITE_PRELOADED_DATA = False 




############ Additional params for the 'intra' case ############
num_repetitions_intra = 3 # (For 'inter', the program will perform leave-one-file-out analysis)
fraction_test_intra = 1.0/3 # Fraction of spectra used for testing in the 'intra' case



############ Default visualization parameters ############

# Specify the color that you want to use for each class
# https://www.w3.org/wiki/CSS/Properties/color/keywords
color_dict = {'SERS':'black',
              'Buffer':'blue',
              'HEK':'green',
              'PC3': 'red',
              'HepG': 'magenta',
              'MCF7':'darkgoldenrod',
              'MDAMB':'teal'}

# Specify if you want to remove outliers when you visualize the data
remove_outlier_visualization = True
percentile_low = 3
percentile_high = 97




############ Machine learning parameters ############

## PCA
num_PCs_kept_reduce_dim = 20
largest_PC_for_visualization = 6

# Specify the PC combinations that you want to plot
# Example: pca_display_list = [[1,2,3], [1,2,4]]
# 1st plot = (PC1,PC2,PC3), 2nd plot (PC1,PC2, PC4),...
# If its length is zero, run all
pca_display_list = [[1,2,3], [1,2,4]]
# pca_display_list = []


## LDA
num_components_kept_LDA_reduce_dim = 20
largest_LDA_component_for_visualization = 6

# Specify the PC combinations that you want to plot
# 1st plot = (LD1,LD2,LD3), 2nd plot (LD1,LD2, LD4),...
# If its length is zero, run all
lda_display_list = [[1,2,3]]