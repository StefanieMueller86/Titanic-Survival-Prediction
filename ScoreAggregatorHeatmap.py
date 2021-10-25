"""
Created on Tue Aug 24 as part of my "Machine Learning" group work course project.
@author: Stefanie Müller

Der ScoreAggregator berechnet den f1 Score eines beliebigen y_pred
und speichert alle eingegebenen Scores in einer Tabelle, die geplottet werden kann.
Die Spaltennamen der Tabelle übergibt man mit dem y_pred.
"""

from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ScoreAggregator:
    """
    Der ScoreAggregator berechnet den f1 Score eines beliebigen y_pred
    und speichert alle Scores in einer Tabelle, die geplottet werden kann

    Benötigt y_true zum Initialisieren.
    f1-Parameter können ebenfalls angegeben werden.
    """

    def __init__(self, y_true, f1_param_average='micro'):
        self.__y_true = y_true
        self.__scores = {}
        self.__f1_param_average = f1_param_average

    def add_score(self, classifier_name, y_pred):
        """
        Gibt f1 score zurück. Best Score: 1, Worst Score: 0
        Fügt f1 score mit angegebenem classifier_name in die Tabelle ein.

        :param classifier_name: String, der als Column-Label in Tabelle dient. Wird überschrieben bei gleichen Angaben.
        :param y_pred: y_pred eines beliebigen Classifieres
        :return: f1_score zu den Eingabedaten
        """
        calculated_f1 = f1_score(self.__y_true, y_pred, average=self.__f1_param_average)

        self.__scores[classifier_name] = [calculated_f1]
        return calculated_f1

    def add_score_manually(self, classifier_name, f1score):
        """
        Fügt manuell einen f1-score hinzu.
        :param classifier_name: String, der als Column-Label in Tabelle dient. Wird überschrieben bei gleichen Angaben.
        :param f1score: Einzutragender f1-Score
        """
        self.__scores[classifier_name] = [f1score]

    def show_score_table(self, plotbreite = 10, plothoehe = 4.5, show=True):
        """
        Plottet die Tabelle gefüllt durch add_score mit allen eingetragenen f1-Scores.
        :param show: gibt bei True den plot in pycharm aus, bei False returned es den plot
        :return gibt den plot zurück, falls show auf False ist
        """
        df = pd.DataFrame.from_dict(self.__scores)

        plt.rcParams["figure.figsize"] = [plotbreite, plothoehe]
        sns.heatmap(df,
                    annot=True,
                    yticklabels = False,
                    cmap='rocket_r',
                    square=True,
                    vmin = 0,
                    vmax = 1,
                    fmt=".2f",
                    cbar_kws = dict(use_gridspec=False, location='top'))
        plt.xticks(rotation=90)
        plt.title('f1 Scores', y=1.7)
        plot = plt

        if show == True:
            plt.show()
        else:
            return plot