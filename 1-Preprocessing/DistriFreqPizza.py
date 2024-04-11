import pandas as pd
import matplotlib.pyplot as plt

def plot_pie_chart(df, column_name):
    # Calcula a contagem de valores únicos na coluna
    counts = df[column_name].value_counts()

    # Calcula a frequência relativa
    relative_freq = counts / counts.sum()

    # Definindo rótulos personalizados para a legenda
    custom_labels = {
        0: 'Não Induzida',
        1: 'Induzida'
    }

    # Obtém os rótulos apropriados usando a legenda personalizada
    labels = [custom_labels.get(val, val) for val in counts.index]

    # Plota o gráfico de pizza
    plt.figure(figsize=(8, 6))
    plt.pie(relative_freq, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f'Distribuição de Frequência Relativa para {column_name}', y=1.05)  # Ajusta a posição do título
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Adiciona a legenda personalizada
    plt.legend(loc='best', bbox_to_anchor=(1, 0.5))

    plt.show()

def main():
    # Faz a leitura do arquivo
    names = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse', 'Resultado']

    input_file = '0-Datasets/HeartAttackModa.csv'

    df = pd.read_csv(input_file, names=names)

    # Chama a função para plotar o gráfico de pizza para uma determinada coluna
    plot_pie_chart(df, 'AnginaInduzExerc')  # Altere 'TipoDorPeito' para o nome da coluna desejada

if __name__ == "__main__":
    main()
