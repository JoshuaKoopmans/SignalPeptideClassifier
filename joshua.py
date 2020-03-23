##############################################
# Authors: Joshua, Michelle                  #
# Description: Signal Peptide dataset parser #
# Date: 13-02-2020                           #
# Last Edited: 23-03-2020                    #
##############################################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, roc_curve, \
    roc_auc_score
import domain.Signal_class as SigClass
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
import epitopepredict as ep

RANDOM_SEED = 314
blosum = ep.blosum62


def main(one_hot=True, nlf=False, blosum=False):
    """
    Central logic of this program: Opening files and parsing resulting in a dictionary with objects.
    """

    proteins = open_and_read_file()

    peptide_dict = parse_proteins(proteins)
    labels = np.zeros(len(peptide_dict["no_sp"]) + len(peptide_dict["sp"]))
    if one_hot:
        n = 21
        encoder = "One-Hot"
    elif nlf:
        n = 18
        encoder = "NLF"
    elif blosum:
        n = 24
        encoder = "BLOSUM62"

    dset = np.zeros([labels.shape[0], 70 * n])
    idx = 0
    for obj in peptide_dict["no_sp"]:
        if one_hot:
            dset[idx] = one_hot_encode(obj.get_protein())
        elif nlf:
            dset[idx] = nlf_encode(obj.get_protein())
        elif blosum:
            dset[idx] = blosum_encode(obj.get_protein())
        idx += 1
    # Add label "1" to sequences with SP
    for obj in peptide_dict["sp"]:
        if one_hot:
            dset[idx] = one_hot_encode(obj.get_protein())
        elif nlf:
            dset[idx] = nlf_encode(obj.get_protein())
        elif blosum:
            dset[idx] = blosum_encode(obj.get_protein())
        labels[idx] = 1
        idx += 1

    X_train, X_test, y_train, y_test = train_test_split(dset, labels, train_size=0.8, random_state=RANDOM_SEED)

    logistic(X_train, X_test, y_train, y_test)
    random_forest(X_train, X_test, y_train, y_test)
    svms(X_train, X_test, y_train, y_test)
    combo(X_train, X_test, y_train, y_test, encoder)


def random_forest(X_train, X_test, y_train, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    regr = RandomForestRegressor(n_estimators=100, random_state=0, min_samples_split=5, min_samples_leaf=4,
                                 max_features="auto", max_depth=30)

    print('=' * 5 + ' Random Forest ' + '=' * 5)
    print(regr.get_params())
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)
    plt.plot(tpr, fpr, thresholds)
    plt.show()
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


def logistic(X_train, X_test, y_train, y_test):
    # Create logistic regression object

    regr = linear_model.LogisticRegression(max_iter=1000000, class_weight="balanced")
    # class_weight={1: np.array(1000000), 0: 1.0}
    print("=" * 5 + " Logistic " + "=" * 5)

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


def combo(X_train, X_test, y_train, y_test, encoder):
    """
    Models (or classifiers) are put into a list and one by one are fitting and predicting using the train and test sets.
    The False Positive Rate (FPR), False Negative  Rate (FNR) and Area Under the Curve (AUC) are computed
    and collected into a dataframe. Using this dataframe, subplots are created for each classifier type.
    The images are saved.

    :param X_train: peptide sequences in train set
    :param X_test: peptide sequences in test set
    :param y_train: actual class of the peptide sequences in train set
    :param y_test: actual class of the peptide sequences in test set
    :param encoder: name of the encoder used
    :return: images of the ROC-curve plots are saved
    """
    classifiers = [LogisticRegression(max_iter=1000000, class_weight="balanced", random_state=RANDOM_SEED),
                   svm.SVC(kernel='linear', probability=True, random_state=RANDOM_SEED),
                   RandomForestClassifier(n_estimators=400, random_state=RANDOM_SEED, min_samples_split=8,
                                          min_samples_leaf=7, max_features="auto", max_depth=50)]

    # Define a result table as a DataFrame
    result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])

    # Train the models and record the results
    for cls in classifiers:
        model = cls.fit(X_train, y_train)
        yproba = model.predict_proba(X_test)[::, 1]

        fpr, tpr, _ = roc_curve(y_test, yproba)
        auc = roc_auc_score(y_test, yproba)

        result_table = result_table.append({'classifiers': cls.__class__.__name__,
                                            'fpr': fpr,
                                            'tpr': tpr,
                                            'auc': auc}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)

    fig = plt.figure(figsize=(8, 6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis with ' + encoder, fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    fig.savefig('multiple_roc_curve_' + encoder + '.png')


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

def one_hot_encode(seq):
    """
       Encoder that puts a "1" if an amino acid is at a position in a matrix where it matches a member of the other
       dimension. This solely creates a machine-readable input for the classifier.

       :param seq: peptide sequence
       :return: vector for each amino acid, matrix for the entire sequence
       """
    codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '*']
    matrix = np.zeros([21, len(seq)])
    for pos, aa in enumerate(seq):
        matrix[codes.index(aa), pos] = 1
    return matrix.reshape(21 * len(seq))


def nlf_encode(seq):
    """
       Encoder based on the NLF, which takes physiochemical properties of amino acids into account.

       :param seq: peptide sequence
       :return: vector for each amino acid, matrix for the entire sequence
       """
    x = pd.DataFrame([nlf[i] for i in seq]).reset_index(drop=True)
    e = x.values.flatten()
    return e


def blosum_encode(seq):
    """
    Encoder based on the BLOSUM62 substitution matrix giving an indication of the conserved parts of a sequence.

    :param seq: peptide sequence
    :return: vector for each amino acid, matrix for the entire sequence
    """
    s = list(seq)
    x = pd.DataFrame([blosum[i] for i in seq]).reset_index(drop=True)
    e = x.values.flatten()
    return e


# Read the NLF matrix (.csv) for the NLF encoder
nlf = pd.read_csv('resources/NLF.csv', index_col=0)

# Run script with One-Hot encoder
main(True, False, False)
# Run script with NLF encoder
main(False, True, False)
# Run script with BLOSUM62 encoder
main(False, False, True)
