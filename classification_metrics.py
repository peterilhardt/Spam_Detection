## Functions for evaluating binary classification models, including calculating various metrics,
## displaying a confusion matrix as a heatmap, and plotting ROC and precision-recall curves

from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, precision_recall_curve, \
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def compute_metrics(model_fit, test_x, test_y):
    """
    Takes a classification model fit using sklearn, a test set prediction matrix, 
    and the test set responses (labels; binary response), and returns a dictionary 
    with the computed accuracy, f1 score, precision, recall, specificity, 
    log-loss, and AUC score (for the ROC curve).
    
    Parameters
    -----------
    model_fit: sklearn classification model fit
    test_x: dataframe of predictor values for test data
    test_y: one-dimensional array or dataframe with test data labels
    
    Returns
    --------
    dictionary with seven calculated model evaluation metrics
    """
    
    pred = model_fit.predict(test_x)
    prob = model_fit.predict_proba(test_x)
    cm = confusion_matrix(test_y, model_fit.predict(test_x))
    
    acc = accuracy_score(test_y, pred)
    logloss = log_loss(test_y, prob)    
    f1 = f1_score(test_y, pred, average = 'binary')
    prec = precision_score(test_y, pred, average = 'binary')
    rec = recall_score(test_y, pred, average = 'binary')
    auc = roc_auc_score(test_y, prob[:,1])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    
    return {'accuracy': acc, 'f1_score': f1, 'precision': prec, 
            'recall': rec, 'specificity': spec, 'log_loss': logloss, 
            'auc': auc}


def print_confusion_matrix(confusion_matrix, class_names, figsize = (7,5), fontsize = 18, file = None):
    """
    Takes a confusion matrix as returned by sklearn.metrics.confusion_matrix and returns it as a heatmap.
    
    Parameters
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order that they index the given confusion matrix.
    figsize: tuple
        A 2-tuple, the first value determining the horizontal size of the output figure, 
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axis labels. Defaults to 18.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix heatmap
    """
    
    df_cm = pd.DataFrame(confusion_matrix, index = class_names, columns = \
        class_names, )
    plt.figure(figsize = figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot = True, fmt = "d", cmap = 'viridis')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, 
                                 ha = 'right', fontsize = fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation = 0, 
                                 fontsize = fontsize)
    plt.ylabel('Actually Spam', fontsize = 15, style = 'italic')
    plt.xlabel('Predicted to be Spam', fontsize = 15, style = 'italic')
    
    if file:
        plt.savefig(file, dpi = 100, bbox_inches = 'tight')
    plt.show()
    plt.close()


def print_roc_curve(model, X, y, file = None):
    """
    Takes a binary classification model fit using sklearn, a prediction matrix, 
    and a response vector (labels), and returns an ROC curve with the AUC score 
    shown in the lower-right.
    
    Parameters
    -----------
    model: sklearn binary classification model fit
    X: dataframe of predictor values
    y: one-dimensional array or dataframe with labels
    
    Returns
    --------
    An ROC curve for the given model and data, with the AUC score shown in the lower-right legend. 
    """
    
    # Get predicted probabilities for test data
    prob = model.predict_proba(X)[:,1]
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y, prob)
    auc = np.round(roc_auc_score(y, prob), 3)
    
    # Plot ROC curve
    plt.figure(figsize = (7, 5))
    plt.plot(fpr, tpr, color = 'darkred', label = 'AUC: {}'.format(auc))
    plt.title("ROC Curve", fontsize = 18, 
              fontweight = 'bold', family = 'serif', y = 1.02)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', style = 'italic', fontsize = 15)
    plt.ylabel('True Positive Rate', style = 'italic', fontsize = 15)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    plt.legend(loc = 'lower right')

    if file:
        plt.savefig(file, dpi = 100, bbox_inches = 'tight')
    plt.show()
    plt.close()


def print_precision_recall_curve(model, X, y, file = None):
    """
    Takes a binary classification model fit using sklearn, a prediction matrix, 
    and a response vector (labels), and returns a precision-recall curve. 
    
    Parameters
    -----------
    model: sklearn binary classification model fit
    X: dataframe of predictor values
    y: one-dimensional array or dataframe with labels
    
    Returns
    --------
    A precision-recall curve for the given model and data.
    """
    
    # Get predicted probabilities for test data
    prob = model.predict_proba(X)[:,1]
    
    # Compute precision-recall curve
    fpr, tpr, _ = precision_recall_curve(y, prob)
    
    # Plot precision-recall curve
    plt.figure(figsize = (7, 5))
    plt.plot(fpr, tpr, color = 'darkred')
    plt.title("Precision-Recall Curve", fontsize = 18, 
              fontweight = 'bold', family = 'serif', y = 1.02)
    plt.xlabel('Recall (Sensitivity)', style = 'italic', fontsize = 15)
    plt.ylabel('Precision', style = 'italic', fontsize = 15)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)

    if file:
        plt.savefig(file, dpi = 100, bbox_inches = 'tight')
    plt.show()
    plt.close()


