import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def main():
    # Faz a leitura do arquivo
    names = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse', 'Resultado']
    features = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse']
    
    input_file = '0-Datasets/HeartAttackModa.csv'
    target = 'Resultado'

    df = pd.read_csv(input_file, names=names)

    # Plotando o gráfico de dispersão
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Resultado'], df['FreqCardioMax'])
    plt.title("Gráfico de Dispersão: Resultado vs. FreqCardioMax")
    plt.xlabel("Resultado")
    plt.ylabel("Frequência Cardíaca Máxima")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()