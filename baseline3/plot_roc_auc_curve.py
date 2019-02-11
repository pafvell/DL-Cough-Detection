#Filipe Barata
import pandas as pd

import matplotlib.pyplot as plt
import sklearn.metrics







if __name__ == "__main__":

    df_rf = pd.read_csv("knn_roc_curve_rf.csv")
    auc_rf = sklearn.metrics.auc(df_rf["fpr"], df_rf["tpr"])

    df_cnn = pd.read_csv("knn_roc_curve_cnn.csv")
    auc_cnn = sklearn.metrics.auc(df_cnn["fpr"], df_cnn["tpr"])

    df_knn = pd.read_csv("knn_roc_curve_knn.csv")
    auc_knn = sklearn.metrics.auc(df_knn["fpr"], df_knn["tpr"])


    plt.figure()
    lw = 2
    plt.plot(df_cnn["fpr"],  df_cnn["tpr"], color='b',
             lw=lw, label='ROC curve CNN (area = %0.2f)' % auc_cnn)

    plt.plot(df_rf["fpr"],  df_rf["tpr"], color='darkorange',
             lw=lw, label='ROC curve RF (area = %0.2f)' % auc_rf)

    plt.plot(df_knn["fpr"],  df_knn["tpr"], color='g',
             lw=lw, label='ROC curve KNN (area = %0.2f)' % auc_knn)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()