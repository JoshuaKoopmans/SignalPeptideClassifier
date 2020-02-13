##############################################
# Authors: Thijs, Michelle, Joshua           #
# Description: Signal Peptide dataset parser #
# Date: 13-02-2020                           #
##############################################

import domain.Signal_class as SigClass
import classifiers.classifiers as Classifiers

import numpy as np
from sklearn.model_selection import train_test_split
import string


def main():
    """
    Central logic of this program: Opening files and parsing resulting in a dictionary with objects.
    """
    proteins = open_and_read_file()
    objects = parse_proteins(proteins)
    
    train_set, test_set = get_data(objects, 0.8, 22)
    
    train, t_labels = split_in_data_labels(train_set)
    test, ts_labels = split_in_data_labels(test_set)
    
    train_pros = convert_string_to_nums(train)
    test_pros = convert_string_to_nums(test)
    
    # print(len(train_pros), len(test_pros))
    # print(len(t_labels), len(ts_labels))

    Classifiers.svc_classifier(train_pros, t_labels, test_pros, ts_labels)


def open_and_read_file():
    """
    Open the training set file and returns individual records.

    :return: List with entries
    """
    with open("../resources/train_set.fasta.txt") as file:
        entries = []
        entry = ""
        for line in file:
            if line.startswith(">"):
                if entry != "":
                    entries.append(entry)
                    entry = ""
            entry += line
        entries.append(entry)  # add last entry
    print(entries[0])
    return entries


def parse_proteins(entries):
    """
    Accepts a list of individual records and utilizes/uses a SigClass object after which objects with no signal and with
    signal peptides are put in respective lists within a dictionary.
    :param entries: List of individual records
    :return: List containing peptide classes
    ------ Edited from:
    :return: Dictionary with lists containing peptide classes with or without signal peptide.
    """
    peptides = [] #{}
    #peptides["no_sp"] = []
    #peptides["sp"] = []
    for entry in entries:
        identifier, sequence, metadata = entry.split("\n")[:-1]
        if "|SP|" in identifier:
            peptides.append(SigClass.Signal_proteins.from_raw(identifier, sequence, metadata))
            # peptides["sp"].append(SigClass.Signal_proteins(identifier, sequence, metadata, True))
        elif "|NO_SP|" in identifier:
            peptides.append(SigClass.Signal_proteins.from_raw(identifier, sequence, metadata))
            # peptides["no_sp"].append(SigClass.Signal_proteins(identifier, sequence, metadata, False))
            
    return peptides
            
            
def convert_string_to_nums(data):
    """
    Temporary conversion to numbers.
    """
    
    #TODO: convert to one hot coding.
    dict_convert = dict(zip(string.ascii_letters,[ord(c)%32 for c in string.ascii_letters]))
    data_list = []
    
    for data_string in data:
        conversion = np.asarray([int(dict_convert[char]) for char in data_string], dtype=np.int32)
        length = len(conversion)
        # TODO: implement trimming
        if length < 70: 
            amount = 70 - length
            conversion = np.append(conversion, np.zeros(amount))
        
        data_list.append(conversion)
      
    return np.array(data_list)
    

def split_in_data_labels(dataset):
    """
    Splits the data from the objects list into a list with the sequence and a list with the label.
    (SP or NO SP)

    """
    data = []
    labels = []

    for signal in dataset:
        data.append(signal.get_protein())
        labels.append(signal.get_has_signal())
        
    return data, labels
    
    
def get_data(all_objects, test_size, seed, shuffle=True):
    """
    Splits the data in 'all_objects' into train and test data.

    """
    train_set, test_set = train_test_split(all_objects, test_size=test_size, random_state=seed, shuffle=shuffle)
    return train_set, test_set


main()
