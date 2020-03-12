##############################################
# Authors: Joshua                            #
# Description: Signal Peptide dataset parser #
# Date: 13-02-2020                           #
# Last Edited: 12-03-2020                    #
##############################################
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

import domain.Signal_class as SigClass
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    """
    Central logic of this program: Opening files and parsing resulting in a dictionary with objects.
    """

    proteins = open_and_read_file()

    peptide_dict = parse_proteins(proteins)
    labels = np.zeros(len(peptide_dict["no_sp"]) + len(peptide_dict["sp"]))
    dset = np.zeros([labels.shape[0], 70 * 21])
    idx = 0
    for obj in peptide_dict["no_sp"]:
        dset[idx] = hotter(obj.get_protein())
        idx += 1
    # Add label "1" to sequences with SP
    for obj in peptide_dict["sp"]:
        dset[idx] = hotter(obj.get_protein())
        labels[idx] = 1
        idx += 1

    X_train, X_test, y_train, y_test = train_test_split(dset, labels, train_size=0.8)
    linear(X_train, X_test, y_train, y_test, False)
    linear(X_train, X_test, y_train, y_test, True)
    random_forest(X_train, X_test, y_train, y_test)
    svms(X_train, X_test, y_train, y_test)

def random_forest(X_train, X_test, y_train, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    regr = RandomForestRegressor(n_estimators=20, random_state=0)
    print('=' * 5 + ' Random Forest ' + '=' * 5)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    print(confusion_matrix(y_test, y_pred.round()))
    print(classification_report(y_test, y_pred.round()))
    print('Accuracy Score :', accuracy_score(y_test, y_pred.round()))

def svms(X_train, X_test, y_train, y_test):
    svc = svm.SVC(kernel='linear')
    print('=' * 5 + ' SVM ' + '=' * 5)
    svc.fit(X_train, y_train)

    predicted = svc.predict(X_test)
    score = svc.score(X_test, y_test)


    print('\nScore ', score)
    print('\nResult Overview\n', metrics.classification_report(y_test, predicted))
    print('\nConfusion matrix:\n', metrics.confusion_matrix(y_test, predicted))


def linear(X_train, X_test, y_train, y_test, linear=True):
    # Create linear or logistic regression object
    if not linear:
        regr = linear_model.LogisticRegression(max_iter=10000000, class_weight={1: np.array(1000000), 0: 1.0})
        print("="*5 + " Logistic " + "="*5)
    else:
        regr = linear_model.LinearRegression()
        print("=" * 5 + " Linear " + "=" * 5)

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred.round()))
    print(set(list(y_pred.round())))

    print('Accuracy Score :', accuracy_score(y_test, y_pred.round()))


def open_and_read_file():
    """
    Open the training set file and returns individual records.

    :return: List with entries
    """
    with open("resources/train_set.fasta.txt") as file:
        entries = []
        entry = ""
        for line in file:
            if line.startswith(">"):
                if entry != "":
                    entries.append(entry)
                    entry = ""
            entry += line
        entries.append(entry)  # add last entry
    return entries


def parse_proteins(entries):
    """
    Accepts a list of individual records and utilizes/uses a SigClass object after which objects with no signal and with
    signal peptides are put in respective lists within a dictionary.
    :param entries: List of individual records
    :return: Dictionary with lists containing peptide classes with or without signal peptide.
    """
    peptides = {}
    peptides["no_sp"] = []
    peptides["sp"] = []
    for entry in entries:
        identifier, sequence, metadata = entry.split("\n")[:-1]
        sequence = (sequence + "*" * (70 - len(sequence)))[:70]
        if "|SP|" in identifier:
            peptides["sp"].append(SigClass.Signal_proteins(identifier, sequence, metadata, True))
        elif "|NO_SP|" in identifier:
            peptides["no_sp"].append(SigClass.Signal_proteins(identifier, sequence, metadata, False))
    return peptides


# All sequences should have the same length. (70)

def hotter(seq="M" * 7):
    codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '*']
    matrix = np.zeros([21, len(seq)])
    for pos, aa in enumerate(seq):
        matrix[codes.index(aa), pos] = 1
    return matrix.reshape(21 * len(seq))




main()
#hotter()


