import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Faz a leitura do arquivo
    names = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse', 'Resultado']

    input_file = '0-Datasets/HeartAttackModa.csv'

    df = pd.read_csv(input_file, names=names)

    QntClasse = 4
    TipoAtrib = 'C'

    if TipoAtrib == 'N':
        # Combinando os dados dos dois grupos para calcular os limites das classes
        combined_df = pd.concat([df[df['Resultado'] == 0], df[df['Resultado'] == 1]])

        # Calculando os limites das classes com base nos dados combinados
        min_value = combined_df['PicoAnterior'].min()
        max_value = combined_df['PicoAnterior'].max()
        bin_width = (max_value - min_value) / QntClasse
        bins = [min_value + i * bin_width for i in range(QntClasse + 1)]

        # Usando os limites de classes calculados para dividir os dados para cada grupo
        SepClasses = pd.cut(df[df['Resultado'] == 0]['PicoAnterior'], bins=bins, include_lowest=True)
        SepClasses1 = pd.cut(df[df['Resultado'] == 1]['PicoAnterior'], bins=bins, include_lowest=True)

        FreqAbs = SepClasses.value_counts().sort_index()
        FreqAbs1 = SepClasses1.value_counts().sort_index()

        # Recuperando os limites dos bins para definir os ticks no gráfico
        bins_ticks = ['({}, {}]'.format(bins[i], bins[i+1]) for i in range(len(bins)-1)]

        bar_width = 0.35
        index = np.arange(len(FreqAbs))

        plt.title('Distribuição de frequência do Pico Anterior por Resultado')

        plt.bar(index, FreqAbs.values, bar_width, color='red', label='Baixas Chances')
        plt.bar(index + bar_width, FreqAbs1.values, bar_width, color='green', label='Altas Chances')

        plt.xlabel('Pico Anterior')
        plt.ylabel('Frequência Absoluta')
        plt.xticks(index + bar_width / 2, bins_ticks)
        plt.legend()

        # Adiciona os rótulos das frequências absolutas em cima de cada barra
        for i, value in enumerate(FreqAbs.values):
            plt.text(i, value, f'{value}', ha='center', va='bottom')
        for i, value in enumerate(FreqAbs1.values):
            plt.text(i + bar_width, value, f'{value}', ha='center', va='bottom')

        plt.show()


    else:
        dfResultado = df[df['Resultado'] == 0]
        dfResultado1 = df[df['Resultado'] == 1]

        coluna_desejada = dfResultado['Sexo']
        coluna_desejada1 = dfResultado1['Sexo']

        FreqAbs = coluna_desejada.value_counts().sort_index()
        FreqAbs1 = coluna_desejada1.value_counts().sort_index()

        bar_width = 0.35
        index = np.arange(len(FreqAbs))

        plt.bar(index, FreqAbs.values, bar_width, color='red', label='Baixas Chances')
        plt.bar(index + bar_width, FreqAbs1.values, bar_width, color='green', label='Altas Chances')

        plt.title('Distribuição de Frequência de Angina por Exercício por Resultado')
        plt.xlabel('Angina Induzida por Exercício')
        plt.ylabel('Frequência Absoluta')
        plt.xticks(index + bar_width / 2, FreqAbs.index.astype(str))
        plt.legend()

        # Adiciona os rótulos das frequências absolutas em cima de cada barra
        for i, value in enumerate(FreqAbs.values):
            plt.text(i, value, f'{value}', ha='center', va='bottom')
        for i, value in enumerate(FreqAbs1.values):
            plt.text(i + bar_width, value, f'{value}', ha='center', va='bottom')

        plt.show()

if __name__ == "__main__":
    main()
