import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Faz a leitura do arquivo
    names = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse', 'Resultado']

    input_file = '0-Datasets/HeartAttackModa.csv'

    df = pd.read_csv(input_file, names=names)

    # Valores fornecidos
    altas_chances = {
        0: 45,
        1: 34.19,
        2: 68.88
    }

    baixas_chances = {
        0: 55,
        1: 65.81,
        2: 30.12
    }

    # Criar um DataFrame com os valores fornecidos
    df = pd.DataFrame({'Altas chances': altas_chances, 'Baixas chances': baixas_chances})

    # Define os intervalos como rótulos do eixo x
    intervalos = ['0','1','2']

    # Criar o gráfico de polígono de frequências
    plt.figure(figsize=(10, 6))
    for coluna in df.columns:
        plt.plot(intervalos, df[coluna], marker='o', label=f'{coluna}')
    plt.title('Polígono de Frequências Relativas entre Declive e Resultado')
    plt.xlabel('Declive')
    plt.ylabel('Frequência Relativa (%)')
    plt.grid(True)
    plt.legend(title='Resultado')
    plt.show()

if __name__ == "__main__":
    main()
