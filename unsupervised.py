## Functions for displaying the results of PCA and cluster analysis as scatterplots colored by groups

from sklearn.decomposition import PCA
from itertools import cycle
import matplotlib.pyplot as plt


def plot_PCA_2D(df, groups, group_labels, file = None, 
                n_components = 2, random_state = None):
    """
    Takes a DataFrame or array, conducts a principal components
    analysis (PCA), and plots the scores of the first two components colored
    by input group labels. Axis labels display the percentage of variance 
    explained by each component.

    Parameters
    ----------
    df: DataFrame or ndarray
    groups: Series or ndarray
        The groups to color the scores by
        length must equal df.shape[0]
    group_labels: one-dimensional array
        The unique labels for the groups
    file: string, default: None
        Filename to save plot output with
    n_components: int, default: 2
        The number of principal components to keep
    random_state: int, default: None

    Returns
    -------
    scatterplot
        scores plot of PC2 vs. PC1 colored by group_labels  
    """

    df = df - df.mean()
    pca = PCA(n_components = n_components, random_state = random_state)
    pca.fit(df)
    scores = pca.transform(df)
    perc_var_explained = pca.explained_variance_ratio_

    colors = cycle(['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'w', 'aqua', 
                    'yellow', 'black', 'brown'])
    plt.figure(figsize = (12, 10))
    for c, label in zip(colors, group_labels):
        plt.scatter(scores[groups == label, 0], scores[groups == label, 1],
                    c = c, label = label, edgecolors = 'gray')
    plt.xlabel('PC1 ({}% Variance)'.format((100 * perc_var_explained[0])\
                .round(2)), fontsize = 15)
    plt.ylabel('PC2 ({}% Variance)'.format((100 * perc_var_explained[1])\
                .round(2)), fontsize = 15)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend()

    if file:
        plt.savefig(file, dpi = 100, bbox_inches = 'tight')
    plt.show()
    plt.close()


def display_clusters(data, model_fit = [], num_clusters = 0, file = None):
    """
    Plots the first 2 columns of an Array as a scatterplot, with the points optionally 
    colored by cluster assignments resulting from a cluster analysis model.

    Parameters
    ----------
    data: Array
    model_fit: cluster model object (default: [])
        Output of a cluster model fit such as sklearn.cluster.KMeans
    num_clusters: int (default: 0)
        Number of clusters to color the points by
    file: str (default: None)
        Filepath to save the plot output to

    Returns
    -------
    scatterplot
        Plot of the first two columns of the input array colored by cluster assignment

    """

    color = ['r', 'g', 'b', 'c', 'm', 'y', 'black', 'orange', 'aqua', 'yellow', 'brown']
    plt.figure(figsize = (12, 10))

    if num_clusters == 0:
        plt.scatter(data[:, 0], data[:, 1], c = color[0])
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)

    else:
        for i in range(num_clusters):
            plt.scatter(data[model_fit.labels_ == i, 0], data[model_fit.labels_ == i, 1], 
                        c = color[i], edgecolors = 'gray')
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)

    if file:
        plt.savefig(file, dpi = 100, bbox_inches = 'tight')
    plt.show()
    plt.close()

