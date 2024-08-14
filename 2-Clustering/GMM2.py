import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def plot_samples(projected, labels, title, ax):    
    u_labels = np.unique(labels)
    for i in u_labels:
        ax.scatter(projected[labels == i, 0], projected[labels == i, 1], label=i,
                   edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('tab10', 10))
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()
    ax.set_title(title)

def main():
    # Load dataset
    input_file = '0-Datasets/HeartAttackModa.csv'
    names = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 
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
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Original shape:", X.shape)
    print("Projected shape:", projected.shape)
    
    # Apply GMM
    gmm = GaussianMixture(n_components=6)  # Adjust n_components as needed
    gmm.fit(projected)
    print("GMM weights:", gmm.weights_)
    print("GMM means:", gmm.means_)
    x = gmm.predict(projected)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot PCA results
    plot_samples(projected, y, 'PCA - Original Labels', axes[0])
    
    # Plot GMM results
    plot_samples(projected, x, 'GMM - Cluster Labels', axes[1])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
