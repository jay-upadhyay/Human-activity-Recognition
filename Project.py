"""
Project.py

Authors: Jay Upadhyay, Sanchitha Seshadri,Sahana Murthy
Description: Human Activity Recgnition using Smartphone Data
it tries 3 different classification models:
    Decision Trees
    Logistic Regression
    MLP Classifier
It also prints following Metrics:
    Mean Squared Error
    Accuracy
    Confusion Matrix
"""
from sklearn import tree
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import itertools
import time
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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

    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_Labels(file1):
    '''
    gets the label names from given filename
    :param file1: Name of file to read from
    :return: list with label names
    '''
    labels=[]
    with open(file1) as fh:
        for line in fh:
            l=line.split()
            labels.append(l[1])
    return labels

def read_file(file_name):
    '''
    Reads the data file
    :param file_name: name of file to read from
    :return:
    '''
    x=file_name
    X = np.loadtxt(x,delimiter=' ')
    #print(X)
    return X
def squared_error(actual, pred):
    '''
    :param actual:
    :param pred:
    :return:
    '''
    return (pred - actual) ** 2
def check(actual, pred):
    '''
    check if actual is equal to redicted
    :param actual: actual value
    :param pred: predicted value
    :return: int 1 or 0
    '''
    if actual==pred:
        return 1
    else:
        return 0
X=read_file("X_train.txt")
Y=read_file("y_train.txt")
X_t=read_file("X_test.txt")
Y_t=read_file("y_test.txt")
#clf = tree.DecisionTreeClassifier()
def model(model):
    return model()
l=[tree.DecisionTreeClassifier,MLPClassifier,linear_model.LogisticRegression]
for t in l:
    t1=time.time()
    clf=model(t)
    clf = clf.fit(X, Y)
    time_taken = time.time() - t1
    #print(len(X_t))
    predicted=clf.predict(X_t)
    #print(len(predicted))
    #print(predicted)
    error=0
    correct=0
    #Log_loss = log_loss(Y_t, predicted)
    for i in range(len(X_t)):
        error+=squared_error(Y_t[i],predicted[i])
        correct+=check(Y_t[i],predicted[i])
    Mse=error/len(X_t)
    conf_mat=confusion_matrix(Y_t, predicted)
    #print("Correct"+str(correct))
    #print("Total"+str(len(X_t)))

    print("Time taken for {} is {}".format(t, time_taken))
    print(clf)
    print("For Model {} mean squared Error is {}".format(t,Mse))
    print("For Model {} Accuracy is {} percent".format(t,(correct/len(X_t))*100))
    print(conf_mat)

    #print("For Model {} Accuracy is {} percent".format(t,Log_loss))
    #print(get_Labels("activity_labels.txt"))
    #plt.figure()
    #plot_confusion_matrix(conf_mat, classes=get_Labels("activity_labels.txt"),title='Confusion matrix, without normalization')
    #plt.show()
#tree.export_graphviz(clf,out_file='tree.dot')
