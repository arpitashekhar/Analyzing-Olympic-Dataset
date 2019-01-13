from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
import numpy as np
from confusion_mat import conf_matrix

def mlp(final_X, final_Y, binary):
    parameters = {'solver': ['lbfgs','sgd','adam'], 'max_iter': [100], 'random_state':[42], 'warm_start':[True]}
    clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1, cv=10)

    clf.fit(final_X, final_Y)

    print(clf.score(final_X, final_Y))
    print(clf.best_estimator_)

    model = clf.best_estimator_
    y_pred = model.predict(final_X)
    print('\nAccuracy: ', accuracy_score(final_Y, y_pred) * 100)
    precision, recall, fscore, support = precision_recall_fscore_support(final_Y, y_pred, average = 'micro')
    print('\nPrecision: ', precision, '\nRecall: ', recall, '\nF-score: ', fscore)
    conf_matrix(final_Y,y_pred, binary)

    '''
    predict = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predict) * 100
    print('\nAccuracy: ', accuracy)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, predict, average = 'micro')
    print('\nPrecision: ', precision, '\nRecall: ', recall, '\nF-score: ', fscore)
    '''
