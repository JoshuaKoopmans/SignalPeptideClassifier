##############################################
# Authors: Thijs, Michelle, Joshua           #
# Description: Signal Peptide dataset parser #
# Date: 13-02-2020                           #
##############################################

import domain.Signal_class as SigClass


def main():
    """
    Central logic of this program: Opening files and parsing resulting in a dictionary with objects.
    """
    proteins = open_and_read_file()
    parse_proteins(proteins)


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
    :return: Dictionary with lists containing peptide classes with or without signal peptide.
    """
    peptides = {}
    peptides["no_sp"] = []
    peptides["sp"] = []
    for entry in entries:
        identifier, sequence, metadata = entry.split("\n")[:-1]
        if "|SP|" in identifier:
            peptides["sp"].append(SigClass.Signal_proteins(identifier, sequence, metadata, True))
        elif "|NO_SP|" in identifier:
            peptides["no_sp"].append(SigClass.Signal_proteins(identifier, sequence, metadata, False))


main()
