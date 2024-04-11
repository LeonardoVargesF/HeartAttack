import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/ClearDataHorseModa.data'
    names = ['Cirurgia',
             'Idade',
             'NumeroHospital',
             'TemperaturaRetal',
             'Pulso',
             'FrequenciaRespiratoria',
             'TemperaturaExtremidades',
             'PulsoPeriferico',
             'MembranasMucosas',
             'TempoRecargaCapilar',
             'Dor',
             'Peristaltismo',
             'DistensaoAbdominal',
             'SondaNasogastrica',
             'RefluxoNasogastrico',
             'PHRefluxoNasogastrico',
             'ExameRetal',
             'Abdome',
             'VolumeCelulasEmbaladas',
             'ProteinaTotal',
             'AparenciaAbdominocentese',
             'ProteinaTotalAbdominocentese',
             'Resultado',
             'LesaoCirurgica',
             'LocalLesao',
             'TipoLesao',
             'SubTipoLesao',
             'CodigoEspecifico',
             'DadosPatologicos']

    features = ['Cirurgia',
             'Idade',
             'NumeroHospital',
             'TemperaturaRetal',
             'Pulso',
             'FrequenciaRespiratoria',
             'TemperaturaExtremidades',
             'PulsoPeriferico',
             'MembranasMucosas',
             'TempoRecargaCapilar',
             'Dor',
             'Peristaltismo',
             'DistensaoAbdominal',
             'SondaNasogastrica',
             'RefluxoNasogastrico',
             'PHRefluxoNasogastrico',
             'ExameRetal',
             'Abdome',
             'VolumeCelulasEmbaladas',
             'ProteinaTotal',
             'AparenciaAbdominocentese',
             'ProteinaTotalAbdominocentese',
             'Resultado',
             'LesaoCirurgica',
             'LocalLesao',
             'TipoLesao',
             'SubTipoLesao',
             'CodigoEspecifico',
             'DadosPatologicos']


    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas
    
    covariancia = df.cov()

    # Cria um heatmap com os valores da correlação
    plt.figure(figsize=(10, 10))
    sns.heatmap(covariancia, annot=True, fmt=".1f", cmap='coolwarm', square=True)
    plt.title("Matriz de Covariancia")
    plt.show()

    
if __name__ == "__main__":
  main()