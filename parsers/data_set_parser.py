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
    return proteins

def parse_proteins(proteins):
    #Dict {no_sp, sp}
    peptides = {}
    peptides["no_sp"] = []
    peptides["sp"] = []
    for protein in proteins:
        
#get_signal_proteins() # list with signal sequences
#get_no_signal_proteins()

main()