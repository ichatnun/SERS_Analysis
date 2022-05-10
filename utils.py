from matplotlib.pyplot import plot, draw, show, ion
import matplotlib.pyplot as plt
import numpy as np
import pdb, os, sys, glob, pickle, random
from webcolors import name_to_rgb, rgb_to_hex
import pandas as pd
from sklearn.metrics import confusion_matrix

from scipy import sparse
from scipy.sparse.linalg import spsolve

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
        curr_num_files_per_class = len(glob.glob(os.path.join(data_base_dir,curr_class,'*.txt')))
        num_files_to_read += curr_num_files_per_class
    
    # Extract common wavenumbers
    for curr_class in class_list:
        filename_list = sorted(glob.glob(os.path.join(data_base_dir,curr_class,'*.txt')))
        
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
            
            if files_already_exist:
                
                # Check the wavenumbers
                temp_df = pd.read_csv(preloaded_raman_shifts_filepath)
                curr_wavenumbers = temp_df['shifts'].values
                del temp_df
                
                if desired_wavenumbers.shape[0] != curr_wavenumbers.shape[0]:
                    print('The preloaded wavenumbers are not the same as the desired ones. Overwrite the preloaded data.')
                    files_already_exist = False
                    
                elif not (desired_wavenumbers == curr_wavenumbers).all():
                    print('The preloaded wavenumbers are not the same as the desired ones. Overwrite the preloaded data.')
                    files_already_exist = False
            
            if (not files_already_exist) or OVERWRITE_PRELOADED_DATA:
                
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
                
                
            # Case: Load the preloaded files  
            else:
                print(preloaded_spectra_filepath, 'already exists, so just load it.')
                f_file = open(preloaded_spectra_filepath, 'rb')
                curr_spectra = pickle.load(f_file)
                f_file.close()
                
                temp_df = pd.read_csv(preloaded_raman_shifts_filepath)
                curr_wavenumbers = temp_df['shifts'].values
                del temp_df
                
                
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


# def plotReducedDimData(x, labels, indices_desired_component, legend_list, colors_list, reduced_dim_method, explained_var = 0):
#     from mpl_toolkits.mplot3d import Axes3D

#     num_components = len(indices_desired_component)
#     num_classes = len(legend_list)

#     if num_components > 3:
#         print("Doesn't support more than 3 dimensions")
#         sys.exit(1)

#     elif num_components == 1:

#         plt.figure()
#         for idx in range(num_classes):
#             indices = np.ravel(labels == idx)
#             data_dim_reduced = x[indices, indices_desired_component[0]]
#             plt.scatter(data_dim_reduced, 0.1*idx*np.ones((data_dim_reduced.shape[0])),
#                         c=colors_list[idx], label=legend_list[idx].lower(), marker='.')
#         plt.ylim((-2,2))

#         if reduced_dim_method in ['pca']:
#             plt.xlabel('PC ' +  str(indices_desired_component[0]+1) +
#                           ' ('+ "{a:2.2f}".format(a=explained_var[indices_desired_component[0]]) + ')')
#         elif reduced_dim_method in ['lda']:
#             plt.xlabel('LD ' + str(indices_desired_component[0] + 1))

#     elif num_components == 2:
#         plt.figure()
#         for idx in range(num_classes):
#             indices = np.ravel(labels == idx)
#             plt.scatter(x[indices, indices_desired_component[0]], x[indices, indices_desired_component[1]],
#                         c=colors_list[idx], label=legend_list[idx].lower(), marker='.')

#         # Adjust the ranges of the axes (ignore outliers)
#         # Adjust the ranges of the axes (ignore outliers)
#         limits_all = []
#         for idx_axis in range(2):

#             ## Initial display = mean +/- 3 sd
#             temp_mean = np.mean(x[:,indices_desired_component[idx_axis]],axis=0)
#             temp_std = np.std(x[:,indices_desired_component[idx_axis]],axis=0)
#             lb_axis = temp_mean - 3 * temp_std
#             ub_axis = temp_mean + 3 * temp_std

#             ## If the 99th percentile of the positive values is less than the current lb, just replace it
#             temp = x[:, indices_desired_component[idx_axis]]
#             temp_pos = temp[temp>=0]
#             ub_axis = min(ub_axis,np.percentile(temp_pos, 99))

#             ## If the 1st percentile of the negative values is higher than the current ub, just replace it
#             temp_neg = temp[temp < 0]
#             lb_axis = max(lb_axis,np.percentile(temp_neg,1))

#             limits_all.append([lb_axis, ub_axis])

#         plt.xlim(limits_all[0])
#         plt.ylim(limits_all[1])

#         # Specify the labels of the axes
#         if reduced_dim_method in ['pca']:
#             plt.xlabel('PC ' +  str(indices_desired_component[0]+1) +
#                           ' ('+ "{a:2.2f}".format(a=explained_var[indices_desired_component[0]]) + ')')
#             plt.ylabel('PC ' +  str(indices_desired_component[1]+1) +
#                           ' ('+ "{a:2.2f}".format(a=explained_var[indices_desired_component[1]]) + ')')
#         elif reduced_dim_method in ['lda']:
#             plt.xlabel('LD ' + str(indices_desired_component[0] + 1))
#             plt.ylabel('LD ' + str(indices_desired_component[1] + 1))

#     elif num_components == 3:
#         ax = Axes3D(plt.figure())
#         for idx in range(num_classes):
#             indices = np.ravel(labels == idx)
#             ax.scatter(x[indices, indices_desired_component[0]], x[indices, indices_desired_component[1]],
#                        x[indices, indices_desired_component[2]], c=colors_list[idx], label=legend_list[idx].lower(),
#                        marker='.')

#         # Adjust the ranges of the axes (ignore outliers)
#         limits_all = []
#         for idx_axis in range(3):

#             ## Initial display = mean +/- 3 sd
#             temp_mean = np.mean(x[:,indices_desired_component[idx_axis]],axis=0)
#             temp_std = np.std(x[:,indices_desired_component[idx_axis]],axis=0)
#             lb_axis = temp_mean - 3 * temp_std
#             ub_axis = temp_mean + 3 * temp_std

#             ## If the 99th percentile of the positive values is less than the current lb, just replace it
#             temp = x[:, indices_desired_component[idx_axis]]
#             temp_pos = temp[temp>=0]
#             ub_axis = min(ub_axis,np.percentile(temp_pos, 99))

#             ## If the 1st percentile of the negative values is higher than the current ub, just replace it
#             temp_neg = temp[temp < 0]
#             lb_axis = max(lb_axis,np.percentile(temp_neg,1))

#             limits_all.append([lb_axis, ub_axis])

#         ax.set_xlim(limits_all[0])
#         ax.set_ylim(limits_all[1])
#         ax.set_zlim(limits_all[2])

#         # Specify the labels of the axes
#         if reduced_dim_method in ['pca']:
#             ax.set_xlabel('PC ' +  str(indices_desired_component[0]+1) +
#                           ' ('+ "{a:2.2f}".format(a=explained_var[indices_desired_component[0]]) + ')')
#             ax.set_ylabel('PC ' +  str(indices_desired_component[1]+1) +
#                           ' ('+ "{a:2.2f}".format(a=explained_var[indices_desired_component[1]]) + ')')
#             ax.set_zlabel('PC ' +  str(indices_desired_component[2]+1) +
#                           ' ('+ "{a:2.2f}".format(a=explained_var[indices_desired_component[2]]) + ')')
#         elif reduced_dim_method in ['lda']:
#             ax.set_xlabel('LD ' + str(indices_desired_component[0] + 1))
#             ax.set_ylabel('LD ' + str(indices_desired_component[1] + 1))
#             ax.set_zlabel('LD ' + str(indices_desired_component[2] + 1))


#     plt.grid()
#     plt.legend()



# def performPCAandSaveResults(spectra_training, spectra_test,labels_training, labels_test, class_list,data_split_method,colors_list, results_path,idx_repetition):
    
#     from sklearn.decomposition import PCA

#     # Train a PCA model
#     clf_PCA = PCA(n_components=20, whiten=True)
#     clf_PCA.fit(spectra_training)

#     # Use the trained PCA model to transform the training and test data
#     spectra_training_PCA = clf_PCA.transform(spectra_training)
#     spectra_test_PCA = clf_PCA.transform(spectra_test)
#     print("PCA: explained var =" + str(clf_PCA.explained_variance_ratio_ * 100))

#     #### See some results for different PCs

#     max_PC = 6
#     indices_desired_component_list = []
#     for idx1 in range(max_PC):
#         for idx2 in np.arange(idx1+1,max_PC):
#             for idx3 in np.arange(idx2+1,max_PC):
#                 indices_desired_component_list.append([idx1,idx2,idx3])

#     for idx_display_set,indices_desired_component in enumerate(indices_desired_component_list):

#         # Display training results
#         plt.figure(1)
#         plotReducedDimData(spectra_training_PCA, labels_training,
#                            indices_desired_component=indices_desired_component,
#                            legend_list=class_list,
#                            colors_list=colors_list,
#                            reduced_dim_method = 'pca',
#                            explained_var = clf_PCA.explained_variance_ratio_*100)

#         title_text = 'Training (rep '+str(idx_repetition+1)+', ' +data_split_method+ ': PCA' 

#         plt.title(title_text)
#         temp_base_results_path = os.path.join(results_path,
#                                        'rep'+str(idx_repetition+1),
#                                        'pca', 
#                                        'training')
#         if not os.path.exists(temp_base_results_path):
#             os.makedirs(temp_base_results_path)
        
#         plt.savefig(fname=os.path.join(temp_base_results_path,
#                                        'set'+ str(idx_display_set+1) + '.jpg'),dpi=300)
#         plt.clf()
#         plt.close()

#         # Display test results
#         plt.figure(1)
#         plotReducedDimData(spectra_test_PCA, labels_test,
#                            indices_desired_component=indices_desired_component,
#                            legend_list=class_list,
#                            colors_list=colors_list,
#                            reduced_dim_method = 'pca',
#                            explained_var = clf_PCA.explained_variance_ratio_*100)

#         title_text = 'Test (rep '+str(idx_repetition+1)+', ' +data_split_method+' : PCA' 

#         plt.title(title_text)        
#         temp_base_results_path = os.path.join(results_path,
#                                        'rep'+str(idx_repetition+1),
#                                        'pca', 
#                                        'test')
#         if not os.path.exists(temp_base_results_path):
#             os.makedirs(temp_base_results_path)
        
#         plt.savefig(fname=os.path.join(temp_base_results_path,
#                                        'set'+ str(idx_display_set+1) + '.jpg'),dpi=300)

#         plt.clf()
#         plt.close()


# def performPCLDAandSaveResults(spectra_training, spectra_test,labels_training, labels_test, class_list,data_split_method,colors_list, results_path, idx_repetition, max_num_components_LDA):

#     from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#     from sklearn.decomposition import PCA

#     # Train a PCA model
#     clf_PCA = PCA(n_components=20, whiten=True)
#     clf_PCA.fit(spectra_training)

#     # Use the trained PCA model to transform the training and test data
#     spectra_training_PCA = clf_PCA.transform(spectra_training)
#     spectra_test_PCA = clf_PCA.transform(spectra_test)

#     # Train an LDA model on the PCA data
#     clf_LDA = LinearDiscriminantAnalysis()
#     clf_LDA.fit(spectra_training_PCA, np.ravel(labels_training))
    
#     # Transform the PCA'ed data
#     spectra_training_LDA = clf_LDA.transform(spectra_training_PCA)
#     spectra_test_LDA = clf_LDA.transform(spectra_test_PCA)
    
#     # Make prediction using the trained classifier
#     labels_training_predicted = clf_LDA.predict(spectra_training_PCA)
#     accuracy_training = np.sum(labels_training_predicted==labels_training)/labels_training.shape[0]*100
#     labels_test_predicted = clf_LDA.predict(spectra_test_PCA)
#     accuracy_test = np.sum(labels_test_predicted==labels_test)/labels_test.shape[0]*100

#     #### See some results for different LDs
#     if max_num_components_LDA == 1:
#         indices_desired_component_list = [[0]]
#     elif max_num_components_LDA == 2:
#         indices_desired_component_list = [[0,1]]
#     elif max_num_components_LDA == 3:
#         indices_desired_component_list = [[0, 1, 2]]
#     else:
#         print('Currently supported 1,2, and 3 dimensions')
#         sys.exit(1)

    
#     for idx_display_set,indices_desired_component in enumerate(indices_desired_component_list):

#         # Display training results
#         plt.figure(1)
#         plotReducedDimData(spectra_training_LDA, labels_training,
#                            indices_desired_component=indices_desired_component,
#                            legend_list=class_list,
#                            colors_list=colors_list,
#                            reduced_dim_method = 'lda')

#         title_text = 'Training (rep '+str(idx_repetition+1)+', ' +data_split_method+': LDA acc = ' + str(accuracy_training) 

#         plt.title(title_text)
#         temp_base_results_path = os.path.join(results_path,
#                                        'rep'+str(idx_repetition+1),
#                                        'pc-lda', 
#                                        'training')
        
#         if not os.path.exists(temp_base_results_path):
#             os.makedirs(temp_base_results_path)
        
#         plt.savefig(fname=os.path.join(temp_base_results_path,
#                                        'set'+ str(idx_display_set+1) + '.jpg'),dpi=300)
            
#         plt.clf()
#         plt.close()

#         # Display test results
#         plt.figure(1)
#         plotReducedDimData(spectra_test_LDA, labels_test,
#                            indices_desired_component=indices_desired_component,
#                            legend_list=class_list,
#                            colors_list=colors_list,
#                            reduced_dim_method = 'lda')

#         title_text = 'Test (rep '+str(idx_repetition+1)+', ' +data_split_method + ': LDA acc = ' + str(accuracy_test) 

#         plt.title(title_text)
#         temp_base_results_path = os.path.join(results_path,
#                                        'rep'+str(idx_repetition+1),
#                                        'pc-lda', 
#                                        'test')
#         if not os.path.exists(temp_base_results_path):
#             os.makedirs(temp_base_results_path)
        
#         plt.savefig(fname=os.path.join(temp_base_results_path,
#                                        'set'+ str(idx_display_set+1) + '.jpg'),dpi=300)

#         plt.clf()
#         plt.close()
        
#     return labels_test_predicted, labels_test
