#Filipe Barata
import pandas as pd

import matplotlib.pyplot as plt
import sklearn.metrics
from matplotlib import rc

rc('text', usetex = True)
rc('font',family='Times New Roman')




if __name__ == "__main__":

    df_rf = pd.read_csv("rf_roc_curve.csv")
    auc_rf = sklearn.metrics.auc(df_rf["fpr"], df_rf["tpr"])

    df_cnn_single = pd.read_csv("cnn_roc_curve_single.csv")
    auc_cnn_single = sklearn.metrics.auc(df_cnn_single["fpr"], df_cnn_single["tpr"])

    df_cnn_ensemble = pd.read_csv("cnn_roc_curve_ensemble.csv")
    auc_cnn_ensemble = sklearn.metrics.auc(df_cnn_ensemble ["fpr"], df_cnn_ensemble["tpr"])


    df_knn = pd.read_csv("knn_roc_curve.csv")
    auc_knn = sklearn.metrics.auc(df_knn["fpr"], df_knn["tpr"])


    plt.figure()
    lw = 2
    plt.plot(df_knn["fpr"], df_knn["tpr"], color='g',
             lw=lw, label=r"\textit{k}-NN  with Hu Moments (AUC = %0.2f)" % auc_knn)


    plt.plot(df_rf["fpr"],  df_rf["tpr"], color='darkorange',
             lw=lw, label=r"RF with PCA (AUC = %0.2f)" % auc_rf)

    plt.plot(df_cnn_single["fpr"], df_cnn_single["tpr"], color='b',
             lw=lw, label=r"CNN with Mel Spec (AUC = %0.2f)" % auc_cnn_single)

    plt.plot(df_cnn_ensemble["fpr"], df_cnn_ensemble["tpr"], color='c',
             lw=lw, label=r"Ensemble CNN  (AUC = %0.2f)" % auc_cnn_ensemble)




    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curves')
    plt.legend(loc="lower right")
    plt.savefig("/home/dfilipe/Desktop/resultsAUC.eps", format='eps',dpi = 1000)
    plt.show()