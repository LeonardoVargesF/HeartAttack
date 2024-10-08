# Initial imports
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from IPython.display import display


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm = np.round(cm, 2)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    




def main():
    input_file = '0-Datasets/HeartAttackMedia.csv'
    names = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse', 'Resultado']
    target = 'Resultado'
    df = pd.read_csv(input_file, names=names)
    df['target'] = target
    
    # Separate X and y data
    X = df[names]
    y = df[target]   
    print("Total samples: {}".format(X.shape[0]))

    # Split the data - 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test  samples: {}".format(X_test.shape[0]))

    # Scale the X data using Z-score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Grid Search for SVM
    parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 10]}
    svm = SVC()
    clf = GridSearchCV(svm, parameters, cv=5)  # 5-fold cross-validation

    # Training using train dataset
    clf.fit(X_train, y_train).best_score_
    results = pd.DataFrame(clf.cv_results_)[['param_kernel', 'split0_test_score', 'split1_test_score']]
    display(results)
    # Display all GridSearchCV results
    


    
    # Predict using test dataset
    y_hat_test = clf.predict(X_test)

    # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Accuracy SVM from sk-learn: {:.2f}%".format(accuracy))
    print("F1 Score SVM from sk-learn: {:.2f}".format(f1))

    # Get test confusion matrix    
    cm = confusion_matrix(y_test, y_hat_test)    
    classes = np.unique(y)    
    plot_confusion_matrix(cm, classes, False, "Confusion Matrix - SVM sklearn")      
    plot_confusion_matrix(cm, classes, True, "Confusion Matrix - SVM sklearn normalized" )  
    plt.show()
    print(sorted(clf.cv_results_.keys()))

    # Cross-validation scores
    cv_scores = cross_val_score(clf, X, y, cv=5)  # 5-fold cross-validation
    print(f'Cross-validation scores: {cv_scores}')
    print(f'Mean cross-validation score: {cv_scores.mean():.2f}')
if __name__ == "__main__":
    main()
