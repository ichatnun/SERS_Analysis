from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.pyplot import plot, draw, show, ion
import matplotlib.pyplot as plt
import numpy as np
import os, pdb
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from config import current_config as config

def getDisplayList(max_comp,display_list):
    
    max_dim_plot = min(3,max_comp)
    
    # Automatically generate the lda_display_list
    if len(display_list) == 0:
        if max_dim_plot == 3:
            desired_component_number_list = []
            for idx1 in range(max_comp):
                for idx2 in np.arange(idx1+1,max_comp):
                    for idx3 in np.arange(idx2+1,max_comp):
                        desired_component_number_list.append([idx1+1,
                                                              idx2+1,
                                                              idx3+1])
                        
        elif max_dim_plot == 2:
            desired_component_number_list = []
            for idx1 in range(max_comp):
                for idx2 in np.arange(idx1+1,max_comp):
                        desired_component_number_list.append([idx1+1,
                                                              idx2+1])
                        
        elif max_dim_plot == 1:
            desired_component_number_list = []
            for idx1 in range(max_comp):
                desired_component_number_list.append([idx1+1])
            
    else:
        desired_component_number_list = display_list
        
    return desired_component_number_list


        
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
    

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# Note that in binary classification, recall of the positive class is also known as “sensitivity”;
# recall of the negative class is “specificity”.
# https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
def computeEvalMetrics(y_true, y_predicted, class_list, results_path):

    classif_report_txt = classification_report(y_true, y_predicted, target_names=class_list)

    with open(os.path.join(results_path, 'eval_metric.txt'), 'w') as f:
        f.write(classif_report_txt)

    eval_metrics_dict = classification_report(y_true, y_predicted, target_names=class_list, output_dict=True)

    return eval_metrics_dict

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
    
    for idx_dim in range(df.shape[1]):
        curr_values = df.iloc[:,idx_dim]
        
        curr_flag_inliers_lb = np.percentile(curr_values, config.percentile_low) <= curr_values
        curr_flag_inliers_ub = curr_values <= np.percentile(curr_values, config.percentile_high)
        
        
        curr_flag_inliers = curr_flag_inliers_lb & curr_flag_inliers_ub
            
        if idx_dim == 0:
            flag_inliers = curr_flag_inliers
        else:
            flag_inliers = flag_inliers&curr_flag_inliers
        
    return flag_inliers
        

def plotScatterPlotly(df, 
                      col_name_list,
                      method_name,
                      results_outfile_path_no_extension,
                      save_html=True, 
                      save_video=False,
                      params=None):

    num_dim = len(col_name_list)
    
    # Remove outliers just for visualization
    if config.remove_outlier_visualization:
        curr_df = df[col_name_list]

        flag_inliers = getInlierFlag(curr_df)
        df_inliers = df[flag_inliers]
    else:
        df_inliers = df

    # Prepare the axis labels 
    axis_labels = []
    if method_name == 'pca':

        for idx_axis in range(num_dim):
            axis_labels.append(col_name_list[idx_axis] + "({a:2.2f}%)".format(a=params['explained_var'][idx_axis]))
    else:
        for idx_axis in range(num_dim):
            axis_labels.append(col_name_list[idx_axis])

    if num_dim == 3:
        fig = px.scatter_3d(x=df_inliers[col_name_list[0]],
            y=df_inliers[col_name_list[1]],
            z=df_inliers[col_name_list[2]],
            color=df_inliers['category'],
            color_discrete_map=config.color_dict)
        
        fig.update_layout(scene = dict(
                            xaxis_title=axis_labels[0],
                            yaxis_title=axis_labels[1],
                            zaxis_title=axis_labels[2]))
    elif num_dim == 2:

        fig = px.scatter(x=df_inliers[col_name_list[0]],
                            y=df_inliers[col_name_list[1]],
                            color=df_inliers['category'],
                            color_discrete_map=config.color_dict)
        fig.update_layout(scene = dict(
                            xaxis_title=axis_labels[0],
                            yaxis_title=axis_labels[1]))
        
    elif num_dim == 1:
        fig = px.scatter(x=df_inliers[col_name_list[0]],
                         y=[0]*df_inliers[col_name_list[0]].shape[0],
                         color=df_inliers['category'],
                         color_discrete_map=config.color_dict)
        fig.update_layout(scene = dict(xaxis_title=axis_labels[0]))       
    else:
        return
        
    fig.update_traces(marker=dict(size=5),
                  selector=dict(mode='markers')) 
    
    # Save the figure
    fig.write_image(results_outfile_path_no_extension+'_plotly.jpg')
    
    # Display the data
    if save_html:
        fig.write_html(results_outfile_path_no_extension+'_plotly.html')
             
def plotScatterMatplotlib(df, 
                          col_name_list,
                          method_name,
                          results_outfile_path_no_extension,
                          save_video=False,
                          params=None):

    num_dim = len(col_name_list)
    
    # Remove outliers just for visualization
    if config.remove_outlier_visualization:
        curr_df = df[col_name_list]

        flag_inliers = getInlierFlag(curr_df)
        df_inliers = df[flag_inliers]
    else:
        df_inliers = df

    # Prepare the axis labels 
    axis_labels = []
    if method_name == 'pca':

        for idx_axis in range(num_dim):
            axis_labels.append(col_name_list[idx_axis] + "({a:2.2f}%)".format(a=params['explained_var'][idx_axis]))
    else:
        for idx_axis in range(num_dim):
            axis_labels.append(col_name_list[idx_axis])
            
    # color_name_for_all_samples = []
    # for curr_class in df_inliers['category']:
    #     color_name_for_all_samples.append(params['color_dict'][curr_class])
            
    if num_dim == 3:
        
        ax = Axes3D(plt.figure())
        
        for curr_class in params['class_list']:
            curr_class_df = df_inliers[df_inliers['category']==curr_class]
            
            ax.scatter(curr_class_df[col_name_list[0]],
                       curr_class_df[col_name_list[1]],
                       curr_class_df[col_name_list[2]],
                       c=params['color_dict'][curr_class],
                       label=curr_class)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
        
    elif num_dim == 2:
        ax = plt.axes()
        for curr_class in params['class_list']:
            curr_class_df = df_inliers[df_inliers['category']==curr_class]
            
            ax.scatter(curr_class_df[col_name_list[0]],
                       curr_class_df[col_name_list[1]],
                       c=params['color_dict'][curr_class],
                       label=curr_class)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])

        
    elif num_dim == 1:
        ax = plt.axes()
        for curr_class in params['class_list']:
            curr_class_df = df_inliers[df_inliers['category']==curr_class]
            
            ax.scatter(curr_class_df[col_name_list[0]],
                       [0]*curr_class_df.shape[0],
                       c=params['color_dict'][curr_class],
                       label=curr_class)
        ax.set_xlabel(axis_labels[0])
    
    plt.grid()
    plt.legend()
    plt.savefig(fname=results_outfile_path_no_extension\
                +'_matplotlib.jpg',dpi=150)
    
    if save_video and (num_dim==3):
        
        plt.rc('animation', html='html5')
        fig = plt.gcf()
        def init():
            return fig,

        def animate(i):
            ax.view_init(elev=10, azim=4*i)
            return fig,

        anim = animation.FuncAnimation(fig, 
                                       animate, 
                                       init_func=init,
                                       frames=90, 
                                       interval=20, 
                                       blit=True)
        
        # print('\t\tExport the .gif file')
        # anim.save(results_outfile_path_no_extension+'.gif',
        #           writer='imagemagick',fps=20, dpi=150)
        # print('\t\tDone: Export the .gif file') 
        
        print('\t\tExport the .gif file')
        writermp4 = animation.FFMpegWriter(fps=20) 
        anim.save(results_outfile_path_no_extension+'.mp4', 
                  writer=writermp4, dpi=150)
        print('\t\tDone: Export the .mp4 file') 
        
        
    plt.clf()
    plt.close()
    
    
    