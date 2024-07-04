import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def main():

    names = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse', 'Resultado']
    features = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse']
    
    target = 'Resultado'
    
    input_file = '0-Datasets/HeartAttackModa.csv'

    df = pd.read_csv(input_file, names=names, usecols = ['PressaoArterialRepouso', 'FreqCardioMax', 'Resultado'])
    print(df.head())

    x = df.loc[:, features].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    normalizedDf = pd.DataFrame(data = x, columns = features)
    normalizedDf = pd.concat([normalizedDf, df[[target]]], axis = 1)

    # PCA projection
    pca = PCA()    
    principalComponents = pca.fit_transform(x)
    print("Explained variance per component:")
    print(pca.explained_variance_ratio_.tolist())

    principalDf = pd.DataFrame(data = principalComponents[:,0:2], 
                               columns = ['principal component 1', 
                                          'principal component 2'
                                          ])

    sns.scatterplot(data = principalDf, x = 'principal component 1', y = 'principal component 2', hue = 'Resultado')
    plt.show();

    # X_train, X_test, y_train, y_test = train_test_split(df[['PressaoArterialRepouso', 'FreqCardioMax']], df[['Resultado']], test_size=0.33, random_state=0)
    # X_train_norm = preprocessing.normalize(X_train)
    # X_test_norm = preprocessing.normalize(X_test)

    # kmeans = KMeans(n_clusters = 2, random_state = 0, n_init='auto')
    # kmeans.fit(X_train_norm)

    # #sns.scatterplot(data = X_train, x = 'PressaoArterialRepouso', y = 'FreqCardioMax', hue = kmeans.labels_)

    # #sns.boxplot(x = kmeans.labels_, y = y_train['Resultado'], palette="Set1")
    
    # #plt.show()

    # #print(silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean'))

    # K = range(2, 8)
    # fits = []
    # score = []


    # for k in K:
    #     # train the model for current value of k on training data
    #     model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(X_train_norm)
        
    #     # append the model to fits
    #     fits.append(model)
        
    #     # Append the silhouette score to scores
    #     score.append(silhouette_score(X_train_norm, model.labels_, metric='euclidean'))

    # #sns.scatterplot(data = X_train, x = 'PressaoArterialRepouso', y = 'FreqCardioMax', hue = fits[0].labels_)
    

    # sns.boxplot(x = fits[3].labels_, y = y_train['Resultado'])
    # plt.show()  

if __name__ == "__main__":
    main()