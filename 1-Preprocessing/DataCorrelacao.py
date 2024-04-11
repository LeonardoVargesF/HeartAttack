import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


def main():
    # Faz a leitura do arquivo
    names = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse', 'Resultado']
   
    input_file = '0-Datasets/HeartAttackMedia.csv'


    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas
    
    correlacao = df.corr()

    # Cria um heatmap com os valores da correlação
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlacao, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Matriz de Correlação")
    plt.show()

    print(correlacao)

if __name__ == "__main__":
  main()