#Filipe Barata
import pandas as pd

import matplotlib.pyplot as plt
import sklearn.metrics







if __name__ == "__main__":

    df = pd.read_csv("knn_roc_curve.csv")
    auc = sklearn.metrics.auc(df["fpr"], df["tpr"])

    plt.figure()
    lw = 2
    plt.plot(df["fpr"],  df["tpr"], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()