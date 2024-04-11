import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Faz a leitura do arquivo
    names = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse', 'Resultado']

    input_file = '0-Datasets/HeartAttackModa.csv'

    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas
                 
    #ShowInformationDataFrame(df,"Dataframe original")

    QntClasse = 4
    TipoAtrib = 'C'

    if TipoAtrib == 'C':

        dfResultado = df[df['Resultado'] == 0]
        SepClasses = pd.cut(dfResultado['ResEletroCardio'], bins=QntClasse)

        FreqAbs = SepClasses.value_counts().sort_index()

        FreqAbs.sort_values()
        print(FreqAbs)

        TotalObs = FreqAbs.sum()

        FreqRel = (FreqAbs / TotalObs)*100

        FreqRel = FreqRel.sort_index()

        print(FreqRel)

        plt.bar(FreqRel.index.astype(str), FreqRel.values)
        plt.xlabel('Classes')
        plt.ylabel('Frequência Absoluta')
    
        plt.show()

    else:
        dfResultado = df[df['Resultado'] == 1]
        coluna_desejada = dfResultado['PressaoArterialRepouso']
        print(coluna_desejada)

        FreqAbs = coluna_desejada.value_counts().sort_index()
        FreqAbs.sort_values()
        print(FreqAbs)

        TotalObs = FreqAbs.sum()

        FreqRel = (FreqAbs / TotalObs)*100

        FreqRel = FreqRel.sort_index()
        print(FreqRel)

        plt.bar(FreqRel.index.astype(str), FreqRel.values)
        plt.xlabel('Classes')
        plt.ylabel('Frequência Relativa')
    
        plt.show()
    
if __name__ == "__main__":
    main()