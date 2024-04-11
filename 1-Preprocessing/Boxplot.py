import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Faz a leitura do arquivo
    names = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse', 'Resultado']

    input_file = '0-Datasets/HeartAttackMedia.csv'

    df = pd.read_csv(input_file,  # Nome do arquivo com dados
                      names=names)  # Nome das colunas
 
    ## Crie um boxplot
    df["Resultado"].unique()

    plt.title("BoxPlot Resultado x TipoDorPeito")

    sns.boxplot(x="Resultado",y="TipoDorPeito",data = df,palette="Set2")

    plt.show()

if __name__ == "__main__":
    main()