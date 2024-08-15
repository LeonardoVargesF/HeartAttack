import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import matplotlib.pyplot as plt

def checkRS(X, y):
    mse_dict = {}  # Mean Squared Error dictionary
    acc_dict = {}  # Accuracy dictionary

    # Itera sobre diferentes valores de max_iter
    for n in range(25, 400, 25):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                 test_size=0.2, random_state=65)
        model_MLP = MLPClassifier(random_state=48,
                                 hidden_layer_sizes=(150, 100, 50),
                                 max_iter=n, activation='relu',
                                 solver='adam')
        model_MLP.fit(X_train, y_train)
        y_pred = model_MLP.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mse_dict.update({n: round(mse, 3)})
        acc_dict.update({n: round(acc * 100, 3)})

    # Mean Squared Error
    lowest = min(mse_dict.values())
    res = [key for key in mse_dict if mse_dict[key] == lowest]
    mse_list = mse_dict.items()
    k, v = zip(*mse_list)
    print("RMSE is lowest at {} for n: {} ".format(round(lowest, 3), res))

    # Plot RMSE values
    plt.figure(figsize=(12, 6))
    plt.plot(k, v, marker='o')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Erro Quadrático Médio")
    plt.title("Erro Quadrático Médio vs Number of Iterations")
    plt.grid(True)
    plt.show()

    # Accuracy
    highest = max(acc_dict.values())
    res1 = [key for key in acc_dict if acc_dict[key] == highest]
    acc_list = acc_dict.items()
    k1, v1 = zip(*acc_list)
    print("Accuracy is highest at {} % for n: {} ".format(highest, res1))
    
    # Plot Accuracy values
    plt.figure(figsize=(12, 6))
    plt.plot(k1, v1, marker='o')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Number of Iterations")
    plt.grid(True)
    plt.show()

def main():
    # Definição dos nomes das colunas
    names = ['Idade', 'Sexo', 'TipoDorPeito', 'PressaoArterialRepouso', 'Colesterol', 'Glicemia',
             'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive',
             'VasosPrincAfetados', 'TesteEstresse', 'Resultado']

    # Carregar o dataset
    input_file = '0-Datasets/HeartAttackModa.csv'
    df = pd.read_csv(input_file, names=names, header=None)

    # Separar características e rótulos
    X = df.drop('Resultado', axis=1)  # Features
    y = df['Resultado']               # Target

    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Chamar a função para verificar os resultados
    checkRS(X_scaled, y)

if __name__ == "__main__":
    main()
