import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Definindo a função KMeans do zero
def KMeans_scratch(x, k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    # Escolhendo aleatoriamente os centróides
    centroids = x[idx, :]  # Passo 1
     
    for _ in range(no_of_iterations): 
        distances = cdist(x, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])
        
        centroids = []
        for i in range(k):
            # Atualizando os centróides calculando a média dos pontos no cluster
            temp_cent = x[points == i].mean(axis=0) 
            centroids.append(temp_cent)
 
        centroids = np.vstack(centroids)  # Centrólides atualizados
         
    return points

def show_digitsdataset(digits):
    fig = plt.figure(figsize=(6, 6))  # tamanho da figura em polegadas
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, str(digits.target[i]))

def plot_samples(projected, labels, title, cmap='tab10'):    
    plt.figure(figsize=(8, 6))
    u_labels = np.unique(labels)
    colors = plt.cm.get_cmap(cmap, len(u_labels))
    
    for i in u_labels:
        plt.scatter(projected[labels == i, 0], projected[labels == i, 1],
                    label=f'Cluster {i}', edgecolor='none', alpha=0.5, cmap=colors)
    
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.legend()
    plt.title(title)
    plt.grid(True)

def elbow_method(X, max_k):
    """
    Calcula a inércia para diferentes valores de k e plota o gráfico do cotovelo.
    """
    distortions = []
    K = range(1, max_k + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, marker = 'o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo para KMeans')
    plt.grid(True)
    plt.show()

def main():
    # Carregar dataset
    input_file = '0-Datasets/HeartAttackModa.csv'
    names = ['Idade', 'Sexo', 'TipoDorPeito', 'PressaoArterialRepouso', 'Colesterol', 'Glicemia', 
             'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 
             'VasosPrincAfetados', 'TesteEstresse', 'Resultado']
    df = pd.read_csv(input_file, names=names)
    target_col = 'Resultado'
    
    # Preparar os dados
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Escalar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Transformar os dados usando PCA
    pca = PCA(2)
    projected = pca.fit_transform(X_scaled)
    
    # Visualizar o dataset original
    plt.figure(figsize=(8, 6))
    plt.scatter(projected[:, 0], projected[:, 1], c=y, cmap=plt.cm.get_cmap('tab10', len(np.unique(y))), alpha=0.5)
    plt.title('Labels Originais')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.colorbar(label='Classe')
    plt.grid(True)
    
    # Aplicar o método do cotovelo para encontrar o número ideal de clusters
    elbow_method(projected, max_k=10)
    
    # Aplicar o KMeans do zero
    k = 6
    labels_scratch = KMeans_scratch(projected, k, 5)
    
    # Visualizar os resultados do KMeans do zero
    plot_samples(projected, labels_scratch, 'Labels dos Clusters KMeans do Zero')

    # Aplicar o KMeans do sklearn
    kmeans = KMeans(n_clusters=k, random_state=0).fit(projected)
    print("Inércia do sklearn KMeans:", kmeans.inertia_)
    score = silhouette_score(projected, kmeans.labels_)    
    print(f"Para n_clusters = {k}, o índice de silhueta é {score:.2f}")

    # Visualizar os resultados do KMeans do sklearn
    plot_samples(projected, kmeans.labels_, 'Labels dos Clusters KMeans do sklearn')

    plt.show()

if __name__ == "__main__":
    main()
