import pandas as pd
import numpy as np

def main():
    # Faz a leitura do arquivo
    names = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse', 'Resultado']
    #features = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse']
    
    input_file = '0-Datasets/heart.csv'

    df = pd.read_csv(input_file, names = names)

    # Introduzindo valores faltantes aleatórios
    missing_ratio = 0.15  # Proporção de valores faltantes desejada (20% neste exemplo)

    for col in df.columns:
        if col != 'Resultado': 
            missing_mask = np.random.choice([True, False], size=len(df), p=[missing_ratio, 1-missing_ratio])

            df.loc[missing_mask, col] = '?'

    output_file = '0-Datasets/heartMissing.csv'
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()