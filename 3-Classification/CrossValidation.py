import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn import svm


def main():

    names = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse', 'Resultado']

    input_file = '0-Datasets/HeartAttackModa.csv'

    df = pd.read_csv(input_file, names=names, header=None)  # Correção para ler corretamente o arquivo CSV sem cabeçalho

    df['target'] = df['Resultado']  # Atribui corretamente a coluna alvo

    # Separate X and y data
    X = df.drop('target', axis=1)
    y = df['target']   
    
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X, y, cv=5)

    print(scores)

if __name__ == "__main__":
    main()
