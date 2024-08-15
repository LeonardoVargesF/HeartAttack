import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
    # Load dataset
    input_file = '0-Datasets/HeartAttackMedia.csv'
    names = ['Idade', 'Sexo', 'TipoDorPeito','PressaoArterialRepouso', 'Colesterol', 'Glicemia', 'ResEletroCardio', 'FreqCardioMax', 'AnginaInduzExerc', 'PicoAnterior', 'Declive', 'VasosPrincAfetados', 'TesteEstresse', 'Resultado']
    target = 'Resultado'
    df = pd.read_csv(input_file, names=names)
    
    # Separate features (X) and target (y)
    X = df[names]
    y = df[target]
    print("Total samples: {}".format(X.shape[0]))

    # Split the data - 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test samples: {}".format(X_test.shape[0]))

    # Scale the data using Z-score normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [1, 10]
    }

    # Create the SVM model and use GridSearchCV for cross-validation
    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=10, scoring='accuracy', return_train_score=True)

    # Fit the model on the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and predict on test data
    best_model = grid_search.best_estimator_
    y_hat_test = best_model.predict(X_test)

    # Evaluate the performance
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Best Parameters found: ", grid_search.best_params_)
    print("Accuracy: {:.2f}%".format(accuracy))
    print("F1 Score: {:.2f}".format(f1))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_hat_test)
    classes = np.unique(y)
    plot_confusion_matrix(cm, classes, False, "Confusion Matrix - SVM")
    plot_confusion_matrix(cm, classes, True, "Normalized Confusion Matrix - SVM")
    plt.show()

    # Display all GridSearchCV results
    results = pd.DataFrame(grid_search.cv_results_)
    print("\nAll cross-validation results:")
    print(results[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']])

if __name__ == "__main__":
    main()
