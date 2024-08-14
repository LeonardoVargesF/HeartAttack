import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def elbow_method(projected, max_clusters=10):
    bic_scores = []
    n_components_range = range(1, max_clusters + 1)
    
    for n in n_components_range:
        gmm = GaussianMixture(n_components=n)
        gmm.fit(projected)
        bic_scores.append(gmm.bic(projected))
    
    # Plot BIC scores
    plt.figure(figsize=(8, 6))
    plt.plot(n_components_range, bic_scores, marker='o')
    plt.xlabel('Número de Clusters')
    plt.ylabel('BIC')
    plt.title('Método do Cotovelo - BIC para GMM')
    plt.xticks(n_components_range)
    plt.grid(True)
    plt.show()

def main():
    # Load dataset
    input_file = '0-Datasets/HeartAttackMedia.csv'
    names = ['Idade', 'Sexo', 'TipoDorPeito', 'PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 
             'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse', 
             'Resultado']
    df = pd.read_csv(input_file, names=names)
    target_col = 'Resultado'
    
    # Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Transform the data using PCA
    pca = PCA(2)
    projected = pca.fit_transform(X_scaled)
    
    # Apply elbow method
    elbow_method(projected, max_clusters=10)  # You can adjust the max_clusters as needed

if __name__ == "__main__":
    main()
