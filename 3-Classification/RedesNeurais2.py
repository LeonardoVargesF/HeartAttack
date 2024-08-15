import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Gráfico da perda (erro)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Erro treino')
    plt.plot(history.history['val_loss'], label='Erro teste')
    plt.title('Histórico de Perda')
    plt.xlabel('Época de treinamento')
    plt.ylabel('Função de custo')
    plt.legend()
    
    # Gráfico da acurácia
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('accuracy', []), label='Acurácia treino')
    plt.plot(history.history.get('val_accuracy', []), label='Acurácia teste')
    plt.title('Histórico de Acurácia')
    plt.xlabel('Época de treinamento')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    names = ['Idade', 'Sexo', 'TipoDorPeito', 'PressaoArterialRepouso', 'Colesterol', 'Glicemia',
         'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive',
         'VasosPrincAfetados', 'TesteEstresse', 'Resultado']

    # Carregar o dataset
    input_file = '0-Datasets/HeartAttackModa.csv'
    df = pd.read_csv(input_file, names=names, header=None)

    # Separar características e rótulos
    X = df.drop('Resultado', axis=1)  # Features
    y = df['Resultado']               # Target

    print("Total samples: {}".format(X.shape[0]))

    # Divide em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test samples: {}".format(X_test.shape[0]))

    # Normaliza os dados de entrada usando Z-score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Cria e treina o modelo de Rede Neural
    model = Sequential()
    model.add(Dense(units=20, activation='relu', input_dim=X_train.shape[1]))  # Ajustar input_dim
    model.add(Dense(units=5, activation='relu'))
    model.add(Dense(units=1, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=85, batch_size=64, validation_data=(X_test, y_test))

    # Plot do histórico de treinamento
    plot_training_history(history)

    # Avaliação do modelo
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Erro no conjunto de teste: {:.2f}'.format(loss))
    print('Acurácia no conjunto de teste: {:.2f}%'.format(accuracy * 100))

if __name__ == "__main__":
    main()
