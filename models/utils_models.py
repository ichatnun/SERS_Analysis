from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import plot, draw, show, ion
import matplotlib.pyplot as plt
import numpy as np
import os, pdb
import plotly.express as px

from config import current_config as config

def performPCA(x_train, x_test):
    from sklearn.decomposition import PCA
    
    # Train a PCA model to reduce the dimension of the data
    clf_PCA = PCA(n_components=config.num_PCs_kept_reduce_dim, whiten=True)
    clf_PCA.fit(x_train)
    
    # Use the trained PCA model to transform the training and test data
    x_train_PCA = clf_PCA.transform(x_train)
    x_test_PCA = clf_PCA.transform(x_test)
    
    return x_train_PCA, x_test_PCA, clf_PCA.explained_variance_ratio_*100


def performPCLDA(x_train, y_train, x_test, class_list):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    x_train_PCA, x_test_PCA,_ = performPCA(x_train, x_test)
            
    num_components = min(len(class_list)-1,config.num_components_kept_LDA_reduce_dim)
    
    # Train an LDA model on the PCA data
    clf_LDA = LinearDiscriminantAnalysis(n_components=num_components)
    clf_LDA.fit(x_train_PCA, np.ravel(y_train))

    # Transform the PCA'ed data
    x_train_PCLDA = clf_LDA.transform(x_train_PCA)
    x_test_PCLDA = clf_LDA.transform(x_test_PCA)
    
    y_test_predicted = clf_LDA.predict(x_test_PCA)
    
    return x_train_PCLDA, x_test_PCLDA, y_test_predicted
    
    
def reduceDim(x_train, y_train, x_test, preproc_type, params):
    if preproc_type == 'pca':
        return performPCA(x_train, x_test)
    elif preproc_type == 'pc-lda':
        return performPCLDA(x_train, y_train, x_test, params['class_list'])
    else:
        return x_train, x_test
    
    
def computeClassifAccuracy(y_true, y_predicted):
     return np.sum(y_predicted==y_true)/y_true.shape[0]*100
    
    
# Print and plot a confusion matrix
# Taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True labels')
    plt.clim(0,sum(cm[0,:]))
    plt.xlabel('Predicted labels')
    
    
def plotAndSaveConfusionMatrix(y_true, y_predicted, class_list,results_path):
    
    ## Compute and save the confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_predicted)
    plot_confusion_matrix(cm, class_list)
    plt.savefig(fname=os.path.join(results_path,'confusion_matrix.jpg'),dpi=300) 
    plt.clf()
    
    
def getInlierFlag(df):
    
    
    for idx_dim in range(3):
        curr_values = df.iloc[:,idx_dim]
        
        curr_flag_inliers_lb = np.percentile(curr_values, config.percentile_low) <= curr_values
        curr_flag_inliers_ub = curr_values <= np.percentile(curr_values, config.percentile_high)
        
        
        curr_flag_inliers = curr_flag_inliers_lb & curr_flag_inliers_ub
            
        if idx_dim == 0:
            flag_inliers = curr_flag_inliers
        else:
            flag_inliers = flag_inliers&curr_flag_inliers
        
    return flag_inliers


# df: samples x 3 dimensions
def plotScatter3D(df, 
                  col_name_for_x, 
                  col_name_for_y, 
                  col_name_for_z,
                  method_name,
                  results_outfile_path_no_extension,
                  save_html=False, 
                  save_video=False,
                  params=None):
    

        # Remove outliers just for visualization
        if config.remove_outlier_visualization:
            curr_df = df[[col_name_for_x,col_name_for_y,col_name_for_z]]
            
            flag_inliers = getInlierFlag(curr_df)
            df_inliers = df[flag_inliers]
        else:
            df_inliers = df
            
        # Prepare the x-axis, y-axis, and z-axis labels
        if method_name == 'pca':
            xaxis_label = col_name_for_x + "({a:2.2f}%)".format(a=params['explained_var'][0])
            yaxis_label = col_name_for_y + "({a:2.2f}%)".format(a=params['explained_var'][1])
            zaxis_label = col_name_for_z + "({a:2.2f}%)".format(a=params['explained_var'][2])
        else:
            xaxis_label = col_name_for_x
            yaxis_label = col_name_for_y
            zaxis_label = col_name_for_z
        
        # Display the data
        if save_html:
            fig = px.scatter_3d(x=df_inliers[col_name_for_x],
                        y=df_inliers[col_name_for_y],
                        z=df_inliers[col_name_for_z],
                        color=df_inliers['category'],
                        color_discrete_map=config.color_dict)

            fig.update_traces(marker=dict(size=5),
                              selector=dict(mode='markers'))

            fig.update_layout(scene = dict(
                                xaxis_title=xaxis_label,
                                yaxis_title=yaxis_label,
                                zaxis_title=zaxis_label))

            fig.write_html(results_outfile_path_no_extension+'.html')
