import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Definição dos nomes das colunas
names = ['Idade', 'Sexo', 'TipoDorPeito', 'PressaoArterialRepouso', 'Colesterol', 'Glicemia',
         'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive',
         'VasosPrincAfetados', 'TesteEstresse', 'Resultado']

# Carregar o dataset
input_file = '0-Datasets/HeartAttackModa.csv'
df = pd.read_csv(input_file, names=names, header=None)

# Exibir as primeiras linhas do dataframe
#print(df.head())

# Separar características e rótulos
X = df.drop('Resultado', axis=1)  # Features
y = df['Resultado']               # Target

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criar e treinar o classificador
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 5), random_state=35)
clf.fit(X_train_scaled, y_train)

# Fazer previsões e avaliar o modelo
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Exibir o valor da perda final
loss = clf.loss_

print(f'Accuracy: {accuracy:.2f}')
#print('Classification Report:')
#print(report)
print(f'Final Loss: {loss:.4f}')
