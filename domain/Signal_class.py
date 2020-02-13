# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:00:06 2020

@author: gebruiker
"""

class Signal_proteins():

    def __init__(self, header, protein, meta, has_signal):
        self.header = header
        self.protein = protein
        self.meta = meta
        self.has_signal = has_signal


    def __str__(self):
        return "{}\n{}\n{}\nHas signal peptide: {}".format(self.header, self.protein, self.meta, self.has_signal)


    @classmethod
    def from_raw(cls, header, protein, meta):
        signal = header.split("|")[2]

        if signal == "NO_SP":
            has_signal = 0
        else:
            has_signal = 1

        return cls(header, protein, meta, has_signal)


    def __set_header(self, header):
        self.header = header

    def __set_protein(self, protein):
        self.protein = protein

    def __set_meta(self, meta):
        self.meta = meta

    def __set_has_signal(self, has_signal):
        self.has_signal = has_signal


    def get_header(self):
        return self.header

    def get_protein(self):
        return self.protein

    def get_meta(self):
        return self.meta

    def get_has_signal(self):
        return self.has_signal
