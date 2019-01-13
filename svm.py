import numpy as npi
import pandas
#from subprocess import check_call
#import seaborn as sns 
#from matplotlib import pyplot as plt 
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from confusion_mat import conf_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def svm(final_X, final_Y, binary):

    # X_train['Sex'],_ = pandas.factorize(X_train['Sex'])
    # X_train['Sport'],_ = pandas.factorize(X_train['Sport'])
    # X_train['NOC'],_ = pandas.factorize(X_train['NOC'])
    # X_train['Host_Country'],_=pandas.factorize(X_train['Host_Country'])
    # y_train['Medal'],_ = pandas.factorize(y_train['Medal'])
    # X_test['Sex'],_ = pandas.factorize(X_test['Sex'])
    # X_test['Sport'],_ = pandas.factorize(X_test['Sport'])
    # X_test['NOC'],_ = pandas.factorize(X_test['NOC'])
    # y_test['Medal'],_ = pandas.factorize(y_test['Medal'])
    # X_test['Host_Country'],_=pandas.factorize(X_test['Host_Country'])
    # params = grid_search(X_train, y_train.values.ravel())

    parameters = {'kernel':('linear', 'rbf','poly', 'sigmoid'),
            'C':[1, 5, 10, 50, 100],
            'gamma':['scale',0.25, .5], 'degree':[1, 2, 4]}
    svc = SVC()
    clf = GridSearchCV(svc, parameters, cv=10)
    clf.fit(final_X, final_Y)
    print(clf.best_params_)
    # with gridsearch - kernel - rbf, c = 5, gamma = scale
    clf = SVC(kernel = 'rbf', gamma = 'scale', c = 5, class_weight='balanced', probability=True)
    clf.fit(final_Y, final_Y)
    predict = clf.predict(final_X)
    accuracy = accuracy_score(final_Y, predict) * 100
    accuracy_dict[param] = accuracy
    print('\nAccuracy: ', accuracy)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, predict, average = 'micro')
    print('\nPrecision: ', precision, '\nRecall: ', recall, '\nF-score: ', fscore)
    conf_matrix(y_test,predicts, binary)
