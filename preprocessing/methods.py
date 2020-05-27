##############################################
# Authors: Joshua                            #
# Description: Signal Peptide dataset parser #
# Date: 13-02-2020                           #
# Last Edited: 25-03-2020                    #
##############################################
from datetime import datetime
import os

import torch

import domain.Signal_class as SigClass
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

RANDOM_SEED = 314


def prepare_classification_data(labels, peptide_dict):
    """
    :param labels: labels for the train set
    :param peptide_dict: dictionary with the peptides for SP/NO_SP for train set
    """
    n = 21
    dset = torch.zeros([labels.shape[0], 70 * n])
    idx = 0
    for obj in peptide_dict["no_sp"]:
        dset[idx] = one_hot_encode(obj.get_protein())
        labels[idx][0] = 1
        idx += 1
    # Add label "1" to sequences with SP
    for obj in peptide_dict["sp"]:
        dset[idx] = one_hot_encode(obj.get_protein())
        labels[idx][1] = 1
        idx += 1

    X_train, X_test, y_train, y_test = train_test_split(dset, labels, train_size=0.8, random_state=RANDOM_SEED)
    return X_train, X_test, y_train, y_test


def parse_datasets():
    """
    Reads the train/test and benchmark files and creates dictionaries with peptides
    :return: labels and peptides for the train/test and benchmark datasets
    """
    proteins = open_and_read_file("resources/train_set.fasta.txt")
    peptide_dict = parse_proteins(proteins)
    labels = torch.zeros(len(peptide_dict["no_sp"]) + len(peptide_dict["sp"]), 2)
    return labels, peptide_dict


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

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives
