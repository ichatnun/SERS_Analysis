from matplotlib.pyplot import plot, draw, show, ion
import matplotlib.pyplot as plt
import numpy as np
import pdb, os, sys, glob, pickle, random
from webcolors import name_to_rgb, rgb_to_hex
import pandas as pd
from sklearn.metrics import confusion_matrix

from scipy import sparse
from scipy.sparse.linalg import spsolve

from config import current_config as config

def get_class_names(curr_path):
    class_abs_path_list = create_dir_list_no_hidden_dir(curr_path)
    
    class_list = []
    for curr_abs_path in class_abs_path_list:        
        class_list.append(os.path.basename(curr_abs_path))
        
    return class_list
    
def create_dir_list_no_hidden_dir(path):
    dir_list = []
    for curr_file in sorted(glob.glob(os.path.join(path, '*'))):
        if os.path.isdir(curr_file):
            dir_list.append(curr_file)
    return dir_list


def createColorList(color_dict,class_list):
    
    colors_list = []
    for curr_class in class_list:
        if curr_class in color_dict.keys():
            curr_color = rgb_to_hex(name_to_rgb(color_dict[curr_class]))
        else:
            curr_color = rgb_to_hex((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            
            # Random a new color that is not already in 'color_dict'
            for idx in range(1000):
                if curr_color in color_dict.keys():
                    curr_color = rgb_to_hex((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
                else:
                    color_dict.update({curr_class:curr_color})
                    break
        colors_list.append(curr_color)
        
    return colors_list, color_dict

# This return a python array of shape (num_rows, num_cols, num_wavenumbers).
# x -> columns, y -> rows
def readSingleChipDataFromTxtFile(filename, 
                                  read_wavenumbers_only = False,
                                  perform_interp = False,
                                  desired_wavenumbers = -1,
                                  REMOVE_BASELINE=False):

    if perform_interp and len(desired_wavenumbers) <= 1:
        print('Please specify desired_wavenumbers')
        sys.exit(1)

    df = pd.read_csv(filename, delimiter='\t+', engine='python')
    # df.info()
    # df.head()

    col_pos = df['#X'].values
    row_pos = df['#Y'].values

    # Find the locations where a new location was acquired based on the x and y positions
    temp = abs(np.diff(col_pos, n=1, axis=0)) + abs(np.diff(row_pos, n=1, axis=0))
    
    # Extract numbers of rows and columns
    num_rows = len(np.unique(row_pos))
    num_cols = len(np.unique(col_pos))
    
    num_spectra = np.sum(temp>0) + 1 # Add one to include the first location which is not captured by np.diff
    if num_spectra != num_rows * num_cols:
        print('Something is wrong in readSingleChipDataFromTxtFile() when extracting row/col positions.')
        sys.exit(1)

    # temp > 0 whenever there is a change in col position or row position
    new_set_indices = np.where(temp > 0)
    new_set_indices = new_set_indices[0] + 1  # Increase the indices by 1's to compensate for the np.diff module

    # Check if all of the spectra are of the same length
    if np.unique(np.diff(new_set_indices, n=1)).shape[0] == 1:
        num_wavenumbers = new_set_indices[0]
    else:
        print('The spectra do not have the same length')
        sys.exit(1)

    wavenumbers = df['#Wave'].iloc[:num_wavenumbers].values

    if read_wavenumbers_only:
        return wavenumbers
    else:
        # Extract the spectra
        temp_spectra = np.reshape(df['#Intensity'].values, (num_rows, num_cols, num_wavenumbers))

        # Interpolation
        if perform_interp:
            from scipy.interpolate import interp1d

            # Interpolate the spectra
            num_wavenumbers = desired_wavenumbers.shape[0]
            spectra = np.zeros((num_rows, num_cols, num_wavenumbers))
            for row in range(num_rows):
                for col in range(num_cols):
                    interp_func = interp1d(wavenumbers, np.squeeze(temp_spectra[row,col,:]), kind='cubic')
                    spectra[row,col,:]  = interp_func(desired_wavenumbers)

            wavenumbers = desired_wavenumbers
        else:
            spectra = temp_spectra

        if REMOVE_BASELINE:
            from BaselineRemoval import BaselineRemoval
            polynomial_degree = 4
        
            for row in range(num_rows):
                for col in range(num_cols):
                    spectra[row, col, :] = BaselineRemoval(spectra[row, col, :]).ModPoly(polynomial_degree)

        return spectra, wavenumbers

    
    

                
# Assume data from the same chip have the same wavenumbers
def getCommonWavenumbers(data_base_dir, class_list):
    print('------ Begin: Get common wavenumbers ------')

    all_wavenumbers = set()
    count = 0
    
    # Count number of files to read
    num_files_to_read = 0
    for curr_class in class_list:
        
        if config.data_split_method in ['inter','intra']:
            curr_num_files_per_class = len(glob.glob(os.path.join(data_base_dir,curr_class,'*.txt')))
            num_files_to_read += curr_num_files_per_class
            
        elif config.data_split_method in ['by-folders']:
            curr_num_files_per_class_train = len(glob.glob(os.path.join(data_base_dir,'train',curr_class,'*.txt')))
            curr_num_files_per_class_test = len(glob.glob(os.path.join(data_base_dir,'test',curr_class,'*.txt')))
            num_files_to_read += (curr_num_files_per_class_train + curr_num_files_per_class_test)
    
    # Extract common wavenumbers
    for curr_class in class_list:
        
        if config.data_split_method in ['inter','intra']:
            filename_list = sorted(glob.glob(os.path.join(data_base_dir,curr_class,'*.txt')))
            
        elif config.data_split_method in ['by-folders']:
            filename_list_train = sorted(glob.glob(os.path.join(data_base_dir,'train',curr_class,'*.txt')))
            filename_list_test = sorted(glob.glob(os.path.join(data_base_dir,'test',curr_class,'*.txt')))
            filename_list = filename_list_train + filename_list_test
        
        if len(filename_list) == 0:
            print(curr_class,'has no files. Exit the program')
            sys.exit(1)
        
        for curr_filename in filename_list:
        
            # Load wavenumbers            
            curr_wavenumbers = readSingleChipDataFromTxtFile(curr_filename,
                                                             read_wavenumbers_only=True,
                                                             perform_interp=False,
                                                             desired_wavenumbers=0)

            # Store the wavenumbers from different chips so that 
            # we make a common list of wavenumbers across all the samples
            all_wavenumbers.add(tuple(curr_wavenumbers))

            # Increment the counter
            count += 1
            print(str(count) + '/' + str(num_files_to_read) + ': ' + curr_filename)
            
    
    # Compute the lowest and highest wavenumbers obtained from all the files. 
    # Also, compute the lowest number of wavenumbers in the files 
    # -> The final wavenumbers will be a vector of length 'min_len', 
    # starting at max_lower_bound and ending at min_upper_bound 
    max_lower_bound = -1
    min_upper_bound = 1000000
    min_len = 1000000
    for curr_wavenumber in all_wavenumbers:
        curr_lb = np.min(curr_wavenumber)
        if curr_lb > max_lower_bound:
            max_lower_bound = curr_lb

        curr_ub = np.max(curr_wavenumber)
        if curr_ub < min_upper_bound:
            min_upper_bound = curr_ub

        curr_len = len(curr_wavenumber)
        if curr_len < min_len:
            min_len = curr_len
    

    # By convention, wave numbers are stored and displayed in descending order
    wavenumbers = np.linspace(start=min_upper_bound, stop=max_lower_bound, num=min_len, endpoint=True)
    
    print('------ End: Get common wavenumbers ------')
    return wavenumbers


# Return a python dictionary with the keys: class1, class2, ...
# spectra['class1'] is a list of Numpy arrays of class1, each of shape num_rows, num_cols, num_wavenumbers
def extractAllSpectra(data_base_dir, 
                      class_list, 
                      desired_wavenumbers, 
                      REMOVE_BASELINE=False,
                      OVERWRITE_PRELOADED_DATA=False):

    print('------ Begin: Extract all the spectra ------')
    all_wavenumbers = set()
    
    count = 0
    num_wavenumbers = desired_wavenumbers.shape[0]
    
    # Count number of files to read
    num_files_to_read = 0
    for curr_class in class_list:
        curr_num_files_per_class = len(glob.glob(os.path.join(data_base_dir,curr_class,'*.txt')))
        num_files_to_read += curr_num_files_per_class


    ## Extract the spectra
    spectra = {}

    for idx_class, curr_class in enumerate(class_list):
            
        filename_list = sorted(glob.glob(os.path.join(data_base_dir,curr_class,'*.txt')))
        
        temp_spectra = []
        for idx_filename, curr_filename in enumerate(filename_list):
            
            # Check if the preloaded data already exist
            if REMOVE_BASELINE:
                preloaded_spectra_filepath = os.path.splitext(curr_filename)[0]+'_spectra_baseline_removed.pkl'
                preloaded_raman_shifts_filepath = os.path.splitext(curr_filename)[0]+'_raman_shifts_baseline_removed.csv'
            else:
                preloaded_spectra_filepath = os.path.splitext(curr_filename)[0]+'_spectra.pkl'
                preloaded_raman_shifts_filepath = os.path.splitext(curr_filename)[0]+'_raman_shifts.csv'
                
            # Case: the preloaded files do not exist or we want to reload them
            files_already_exist = os.path.exists(preloaded_spectra_filepath) and os.path.exists(preloaded_raman_shifts_filepath)
            
            # If the files exist, check if the raman shifts are the same as the desired ones. If not, treat them as "non-existing"
            if files_already_exist:
                
                # Check the wavenumbers
                temp_df = pd.read_csv(preloaded_raman_shifts_filepath)
                curr_wavenumbers = temp_df['shifts'].values 
                del temp_df
                
                if desired_wavenumbers.shape[0] != curr_wavenumbers.shape[0]:
                    print('The preloaded number of raman shifts are not of the same length as the desired ones. Overwrite the preloaded data.')
                    files_already_exist = False
                    
                elif not (np.isclose(curr_wavenumbers,desired_wavenumbers,0,0.0001).all()):
                    print('The preloaded raman shifts are not the same as the desired ones. Overwrite the preloaded data.')
                    files_already_exist = False
            
            
            if files_already_exist and (not OVERWRITE_PRELOADED_DATA):
                # Files exist, but not overwrite
                print(preloaded_spectra_filepath, 'already exists, so just load it.')
                f_file = open(preloaded_spectra_filepath, 'rb')
                curr_spectra = pickle.load(f_file)
                f_file.close()
                
                temp_df = pd.read_csv(preloaded_raman_shifts_filepath)
                curr_wavenumbers = temp_df['shifts'].values
                del temp_df
                
            else: 
                # Files exist but want to overwrite or files don't exist
                if files_already_exist:
                    print(preloaded_spectra_filepath, 'already exists, but the user wants to reload and overwrite it.')
                
                # Read data
                curr_spectra, curr_wavenumbers = readSingleChipDataFromTxtFile(curr_filename,
                                                                               read_wavenumbers_only=False,
                                                                               perform_interp=True,
                                                                               desired_wavenumbers=desired_wavenumbers,
                                                                               REMOVE_BASELINE=REMOVE_BASELINE)
                
                # Save the data
                f_file = open(preloaded_spectra_filepath, 'wb')
                pickle.dump(curr_spectra, f_file)
                f_file.close()

                df_out = pd.DataFrame(curr_wavenumbers, columns=['shifts'])
                df_out.to_csv(preloaded_raman_shifts_filepath, index=False, index_label='shifts')
                del df_out                
                
            # Check the wavenumbers; They should all be the same
            all_wavenumbers.add(tuple(curr_wavenumbers))
            if len(all_wavenumbers) > 1:
                print('Found different sets of wavenumbers.')
                print('The problem might have arose because there are some data that have been preloaded with different wavenumbers than the ones that have been determined on the fly (unpreloaded data)')
                print('Consider setting the OVERWRITE_PRELOADED_DATA variable to True')

                sys.exit(1)

            # No problem encountered -> Add the spectra of the current file to the whole thing
            temp_spectra.append(curr_spectra)
                
            # Increment the counter
            count += 1
            print('Done processing '+ str(count) + '/' + str(num_files_to_read) + ': ' + curr_filename + '\n')
        
        spectra[curr_class] = temp_spectra
        
    print('------ End: Extract all the spectra ------')
    return spectra, curr_wavenumbers


# spectra: the format returned by extractAllSpectra
def loadSpectraAndLabels(spectra, class_list):
    
    # For each class
    for idx_class, curr_class in enumerate(class_list):
        spectra_curr_class = spectra[curr_class]

        # For each file in the current class
        for idx_file in range(len(spectra_curr_class)):
            spectra_curr_file = spectra[curr_class][idx_file]
            _,_, num_wavenumbers = spectra_curr_file.shape

            spectra_curr_file = np.reshape(spectra_curr_file,(-1, num_wavenumbers)) 
            labels_curr_file = idx_class*np.ones(spectra_curr_file.shape[0]) # (num_spectra,)

            if idx_class == 0 and idx_file == 0:
                spectra_all = spectra_curr_file
                labels_all = labels_curr_file
            else:
                spectra_all = np.concatenate((spectra_all,spectra_curr_file),axis=0)
                labels_all = np.concatenate((labels_all,labels_curr_file),axis=0)
    
    return spectra_all, labels_all

def trainTestSplitting(data_split_method, 
                       spectra, 
                       class_list, 
                       fraction_test, # Get used for the 'intra' case
                       random_seed = None, # Get used for the 'intra' case
                       idx_target_file_for_inter_test=-1):  # Get used for the 'inter' case
    
    ## spectra: a dictionary with the keys corresponding to the classes 
    # (spectra['class1'] gives you a Python list where each element corresponds to a single file
    # and has a shape of  (num_rows, num_cols, num_wavenumbers) 

    if data_split_method in ['intra']:
        from sklearn.model_selection import train_test_split
        
        spectra_all, labels_all = loadSpectraAndLabels(spectra, class_list)

        # Split the data
        spectra_training, spectra_test, labels_training, labels_test=train_test_split(spectra_all,labels_all,test_size =fraction_test,random_state=random_seed,shuffle=True,stratify=labels_all)
        
    # Take one file per class for testing, and the rest will be used as training data. So, it could be imbalance
    elif data_split_method in ['inter']:
        
        count_training = 0
        count_test = 0
        
        for idx_class, curr_class in enumerate(class_list):
            spectra_curr_class = spectra[curr_class]
            
            # For each file in the current class
            num_files_curr_class = len(spectra_curr_class)
            for idx_file in range(num_files_curr_class):
                
                if idx_target_file_for_inter_test >= num_files_curr_class:
                    print('The given idx_target_file_for_inter_test cannot be greater than or equal to the number of files that you have for the current class.')
                    sys.exit(1)
                    
                spectra_curr_file = spectra_curr_class[idx_file]
                _,_, num_wavenumbers = spectra_curr_file.shape

                spectra_curr_file = np.reshape(spectra_curr_file,(-1, num_wavenumbers)) 
                labels_curr_file = idx_class*np.ones(spectra_curr_file.shape[0]) # (num_spectra,)

                # Test data
                if idx_file == idx_target_file_for_inter_test:
                    if count_test == 0:
                        spectra_test = spectra_curr_file
                        labels_test = labels_curr_file
                    else:
                        spectra_test = np.concatenate((spectra_test,spectra_curr_file),axis=0)
                        labels_test = np.concatenate((labels_test,labels_curr_file),axis=0)
                    count_test += 1

                # Training data
                else:
                    if count_training == 0:
                        spectra_training = spectra_curr_file
                        labels_training = labels_curr_file                        
                    else:
                        spectra_training = np.concatenate((spectra_training,spectra_curr_file),axis=0)
                        labels_training = np.concatenate((labels_training,labels_curr_file),axis=0)
                    count_training += 1
    else:
        sys.exit(1)


    return spectra_training, spectra_test, labels_training, labels_test


def saveSingleEvaluationMetrics_AllMethods(metric,
                                           metric_name,
                                           num_repetitions,
                                           ML_classif_method_list,
                                           color_list_for_bar_plots,
                                           results_path):

    mean_metric = np.mean(metric, axis=0)
    std_metric = np.std(metric, axis=0)
    stats = np.concatenate((metric, np.expand_dims(mean_metric, axis=0)), axis=0)
    stats = np.concatenate((stats, np.expand_dims(std_metric, axis=0)), axis=0)

    df = pd.DataFrame(stats, columns=ML_classif_method_list)

    df_index_temp = []
    for idx_rep in range(num_repetitions):
        df_index_temp.append('Rep ' + str(idx_rep + 1))
    df_index_temp.append('Mean')
    df_index_temp.append('Std')
    df.index = df_index_temp

    df.to_csv(os.path.join(results_path, metric_name+'_test_reps.csv'))

    # Create and save the classification accuracy bar plot
    plt.figure()
    pos = range(len(ML_classif_method_list))
    plt.bar(pos, mean_metric, yerr=std_metric, color=color_list_for_bar_plots)
    plt.xticks(pos, ML_classif_method_list, rotation=90)
    plt.xlabel('ML methods')
    plt.ylabel(metric_name)
    plt.grid()
    plt.savefig(fname=os.path.join(results_path, metric_name+'_test_bar_plot.jpg'), dpi=300, bbox_inches='tight')
    plt.show()
