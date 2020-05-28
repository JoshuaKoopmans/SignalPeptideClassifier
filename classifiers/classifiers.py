##############################################
# Authors: Joshua, Michelle, Thijs           #
# Description: Signal Peptide dataset parser #
# Date: 13-02-2020                           #
# Last Edited: 25-03-2020                    #
##############################################
from datetime import datetime
import os

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, \
    roc_auc_score
import domain.Signal_class as SigClass
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import epitopepredict as ep
from model import ClassifierModel
import matplotlib.pyplot as plt

RANDOM_SEED = 314
blosum = ep.blosum62
benchmark = False #modify this parameter to True to validate with benchmark set

def main():
    """
    Central logic of this program: Opening files and parsing resulting into a dictionary with objects.
    """
    labels, labels_test, peptide_dict, peptide_dict_test = parse_datasets()
    encoders = ["one_hot", "nlf", "blosum"]
    encoders = [encoders[0]]
    for encoder in encoders:
        X_train, X_test, y_train, y_test, dset_test, labels_test, input_dim = prepare_classification_data(labels, labels_test, peptide_dict, peptide_dict_test, encoder)
        if benchmark:
            print("Validating with benchmark set")

        elif not benchmark:
            print("Validating with splitted train set")
            seqs = X_train.view(len(X_train), 21, 70)
            labels = y_train
            net = ClassifierModel(input_dim)
            epochs = 30


            losses = []
            optimizer = torch.optim.Adam(net.parameters())
            for epoch in range(epochs):
                print("Epoch " + str(epoch))
                epoch_loss = 0
                correct = 0
                shuffle = torch.randperm(len(seqs))
                seqs = seqs[shuffle]
                labels = labels[shuffle]
                predicted_list = []
                total_labels = []
                for minibatchindex in range(0, len(seqs), 32):
                    mb_data = seqs[minibatchindex:minibatchindex + 32]
                    mb_labels = labels[minibatchindex:minibatchindex + 32]

                    prediction = net(mb_data)

                    predicted_list.append(prediction[0][0][0].flatten().detach().numpy())
                    total_labels.append(mb_labels[0].detach().numpy())
                    # print(prediction.shape)
                    # print(mb_labels.shape)
                    #loss = ((prediction-mb_labels)**2).sum()
                    loss = F.binary_cross_entropy(prediction, mb_labels)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # print((prediction == mb_labels).sum())
                    # print((prediction != mb_labels).sum())

                    correct += (prediction.argmax() == mb_labels).float().sum()
                    accuracy = 100 * correct / len(mb_data)
                    epoch_loss += float(loss.detach())
                    # print(prediction)
                    # print(labels_test)

                losses.append(epoch_loss)
                print(' Epoch loss: %.20f, Accuracy %.20f' % (epoch_loss / 100, accuracy))


            net.eval()
            output = net(seqs).view(-1, 1)
            # print(output)
            # for i, n in enumerate(output):
            #     print(n, labels[i])
            fpr, tpr, _ = roc_curve(total_labels, predicted_list)
            auc = roc_auc_score(total_labels, predicted_list)
            fig = plt.figure(figsize=(8, 6))

            plt.plot(fpr, tpr, label="AUC={:.3f}".format(auc))
            plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
            plt.xticks(np.arange(0.0, 1.1, step=0.1))
            plt.xlabel("False Positive Rate", fontsize=15)
            plt.yticks(np.arange(0.0, 1.1, step=0.1))
            plt.ylabel("True Positive Rate", fontsize=15)
            plt.title('ROC Curve Analysis with ' + encoder, fontweight='bold', fontsize=15)
            plt.legend(prop={'size': 13}, loc='lower right')
            plt.savefig("roc.png")

            plt.figure(2)
            plt.plot(range(epochs), losses)
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            # plt.plot(fpr_list, tpr_list)
            #
            #plt.show()
            plt.savefig("mygraph.png")




def prepare_classification_data(labels, labels_test, peptide_dict, peptide_dict_test, encoder):
    """
    :param labels: labels for the train set
    :param labels_test:  labels for the test/ benchmark set
    :param peptide_dict: dictionary with the peptides for SP/NO_SP for train set
    :param peptide_dict_test: dictionary with the peptides for SP/NO_SP for test/benchmark set
    :param encoder: encoder used
    """
    if encoder == "one_hot":
        n = 21
        #encoder = "One-Hot"
    elif encoder == "nlf":
        n = 18
        #encoder = "NLF"
    elif encoder == "blosum":
        n = 24
        #encoder = "BLOSUM62"
    dset = torch.zeros([labels.shape[0], 70 * n])
    idx = 0
    for obj in peptide_dict["no_sp"]:
        if encoder == "one_hot":
            dset[idx] = one_hot_encode(obj.get_protein())
        elif encoder == "nlf":
            dset[idx] = nlf_encode(obj.get_protein())
        elif encoder == "blosum":
            dset[idx] = blosum_encode(obj.get_protein())
        #labels[idx][0] = 1
        idx += 1
    # Add label "1" to sequences with SP
    for obj in peptide_dict["sp"]:
        if encoder == "one_hot":
            dset[idx] = one_hot_encode(obj.get_protein())
        elif encoder == "nlf":
            dset[idx] = nlf_encode(obj.get_protein())
        elif encoder == "blosum":
            dset[idx] = blosum_encode(obj.get_protein())
        #labels[idx][1] = 1
        labels[idx] = 1
        idx += 1
    dset_test = np.zeros([labels_test.shape[0], 70 * n])
    idx = 0
    for obj in peptide_dict_test["no_sp"]:
        if encoder == "one_hot":
            dset_test[idx] = one_hot_encode(obj.get_protein())
        elif encoder == "nlf":
            dset_test[idx] = nlf_encode(obj.get_protein())
        elif encoder == "blosum":
            dset_test[idx] = blosum_encode(obj.get_protein())
        idx += 1
    # Add label "1" to sequences with SP
    for obj in peptide_dict_test["sp"]:
        if encoder == "one_hot":
            dset_test[idx] = one_hot_encode(obj.get_protein())
        elif encoder == "nlf":
            dset_test[idx] = nlf_encode(obj.get_protein())
        elif encoder == "blosum":
            dset_test[idx] = blosum_encode(obj.get_protein())
        labels_test[idx] = 1
        idx += 1
    X_train, X_test, y_train, y_test = train_test_split(dset, labels, train_size=0.8, random_state=RANDOM_SEED)
    return X_train, X_test, y_train, y_test, dset_test, labels_test, n


def parse_datasets():
    """
    Reads the train/test and benchmark files and creates dictionaries with peptides
    :return: labels and peptides for the train/test and benchmark datasets
    """
    proteins = open_and_read_file("../resources/train_set.fasta.txt")
    proteins_test = open_and_read_file("../resources/benchmark_set.fasta.txt")
    peptide_dict = parse_proteins(proteins)
    peptide_dict_test = parse_proteins(proteins_test)
    labels = torch.zeros(len(peptide_dict["no_sp"]) + len(peptide_dict["sp"]))
    labels_test = torch.zeros(len(peptide_dict["no_sp"]) + len(peptide_dict["sp"]))
    return labels, labels_test, peptide_dict, peptide_dict_test


def train_multiple_classifiers(X_train, y_train, encoder, dset_test, labels_test):
    """
    Models (or classifiers) are put into a list and one by one are fitting and predicting using the train and test sets.
    The False Positive Rate (FPR), False Negative  Rate (FNR) and Area Under the Curve (AUC) are computed
    and collected into a dataframe. Using this dataframe, subplots are created for each classifier type.
    The images are saved.

    :param X_train: peptide sequences in train set
    :param X_test: peptide sequences in test set
    :param dset_test: actual class of the peptide sequences in benchmark set
    :param labels_test: actual class of the peptide sequences in benchmark set
    :param encoder: name of the encoder used
    """
    classifiers = [LogisticRegression(max_iter=1000000, class_weight="balanced", random_state=RANDOM_SEED),
                   svm.SVC(kernel='linear', probability=True, class_weight="balanced", random_state=RANDOM_SEED),
                   RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=RANDOM_SEED, min_samples_split=8,
                                          min_samples_leaf=7, max_features="auto", max_depth=50)]

    # Define a result table as a DataFrame
    result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])

    # Train the models and record the results
    for cls in classifiers:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Start Time =", current_time)

        model = cls.fit(X_train, y_train)
        yproba = model.predict_proba(dset_test)[::, 1]
        y_pred = cls.predict(dset_test)

        fpr, tpr, _ = roc_curve(labels_test, yproba)
        auc = roc_auc_score(labels_test, yproba)

        result_table = result_table.append({'classifiers': cls.__class__.__name__,
                                            'fpr': fpr,
                                            'tpr': tpr,
                                            'auc': auc}, ignore_index=True)

        print("=" * 5 + cls.__class__.__name__ + "+" + encoder + "=" * 5)
        print("Confusion matrix: ")
        print(confusion_matrix(labels_test, y_pred.round()))
        print('Accuracy Score :', accuracy_score(labels_test, y_pred.round()))

    plot_multple_roc(encoder, result_table)


def plot_multple_roc(encoder, result_table):
    """
    The roc curve of the different classifiers are plotted in one graph
    :param encoder: name of the encoder used
    :param result_table: table with the output from the differen classifiers
    :return: saved images with the plots
    """
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
    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if benchmark:
        filename = os.path.join(results_dir ,'multiple_roc_curve_' + encoder + '.png')
    elif not benchmark:
        filename = os.path.join(results_dir, 'multiple_roc_curve_' + encoder + '_benchmark.png')
    fig.savefig(filename)
    print("Saved " + filename)


def open_and_read_file(filename):
    """
    Open the training set file and returns individual records.

    :return: List with entries
    """
    with open(filename) as file:
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
    matrix = torch.zeros([21, len(seq)])
    for pos, aa in enumerate(seq):
        matrix[codes.index(aa), pos] = 1
    return matrix.reshape(21 * len(seq))


def nlf_encode(seq):
    """
       Encoder based on the NLF, which takes physiochemical properties of amino acids into account.

       :param seq: peptide sequence
       :return: vector for each amino acid, matrix for the entire sequence
       """
    x = pd.DataFrame([nlf[i] for i in seq])
    e = x.values.flatten()
    return torch.from_numpy(e)


def blosum_encode(seq):
    """
    Encoder based on the BLOSUM62 substitution matrix giving an indication of the conserved parts of a sequence.

    :param seq: peptide sequence
    :return: vector for each amino acid, matrix for the entire sequence
    """
    s = list(seq)
    x = pd.DataFrame([blosum[i] for i in seq]).reset_index(drop=True)
    e = x.values.flatten()
    return torch.from_numpy(e)


# Read the NLF matrix (.csv) for the NLF encoder
nlf = pd.read_csv('../resources/NLF.csv', index_col=0)

main()

