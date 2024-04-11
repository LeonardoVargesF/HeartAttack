import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():
    # Faz a leitura do arquivo
    names = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse', 'Resultado']

    input_file = '0-Datasets/HeartAttackModa.csv'

    df = pd.read_csv(input_file, names=names)

    QntClasse = 4
    TipoAtrib = 'C'

    if TipoAtrib == 'N':
        min_value = df['Colesterol'].min()
        max_value = df['Colesterol'].max()
        bin_width = (max_value - min_value) / QntClasse
        bins = [min_value + i * bin_width for i in range(QntClasse + 1)]

        SepClasses = pd.cut(df['Colesterol'], bins=bins, include_lowest=True)
        print(SepClasses)

        FreqAbs = SepClasses.value_counts().sort_index()

        TotalObs = FreqAbs.sum()

        FreqRel = (FreqAbs / TotalObs) * 100

        FreqRel = FreqRel.sort_index()

        print(FreqRel)

        plt.bar(FreqRel.index.astype(str), FreqAbs.values)  # Mudança aqui
        plt.title('Distribuição de frequência do Colesterol')
        plt.xlabel('Colesterol')
        plt.ylabel('Frequência Absoluta')

        # Adiciona os valores das frequências absolutas em cima de cada barra
        for i, value in enumerate(FreqAbs.values):
            plt.text(i, value, f'{value}', ha='center', va='bottom')

        plt.show()

    else:
        coluna_desejada = df['TipoDorPeito']
        print(coluna_desejada)

        FreqAbs = coluna_desejada.value_counts().sort_index()

        TotalObs = FreqAbs.sum()

        FreqRel = (FreqAbs / TotalObs) * 100

        FreqRel = FreqRel.sort_index()

        print(FreqRel)

        plt.bar(FreqRel.index.astype(str), FreqAbs.values)  # Mudança aqui
        plt.title('Distribuição de Frequência do Teste de Estresse')
        plt.xlabel('Teste de Estresse')
        plt.ylabel('Frequência Absoluta')

        # Adiciona os valores das frequências absolutas em cima de cada barra
        for i, value in enumerate(FreqAbs.values):
            plt.text(i, value, f'{value}', ha='center', va='bottom')

        plt.show()

if __name__ == "__main__":
    main()
