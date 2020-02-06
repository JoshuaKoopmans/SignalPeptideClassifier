#open_and_read_file() # return string
#parse_proteins() # dict NO_SP and SP

def main():
    open_and_read_file()

def open_and_read_file():
    with open("/home/michelle/Downloads/SignalP/train_set.fasta") as file: #TODO: include file in git
        proteins = []
        protein = ""
        for line in file:
            if line.startswith(">"):
                if protein != "":
                    proteins.append(protein)
                    protein = ""
            protein += line
        proteins.append(protein) #add last protein



#get_signal_proteins() # list with signal sequences
#get_no_signal_proteins()

main()