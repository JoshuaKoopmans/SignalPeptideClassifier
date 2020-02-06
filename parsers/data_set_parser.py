#open_and_read_file() # return string
#parse_proteins() # dict NO_SP and SP

def main():
    proteins = open_and_read_file()
    parse_proteins(proteins)
def open_and_read_file():
    with open("../resources/train_set.fasta.txt") as file: #TODO: include file in git
        proteins = []
        protein = ""
        for line in file:
            if line.startswith(">"):
                if protein != "":
                    proteins.append(protein)
                    protein = ""
            protein += line
        proteins.append(protein) #add last protein
    print(proteins[0])
    return proteins

def parse_proteins(proteins):
    #Dict {no_sp, sp}
    peptides = {}
    peptides["no_sp"] = []
    peptides["sp"] = []
    for protein in proteins:
        identifier, sequence, metadata = protein.split("\n")[:-1]
        if "|SP|" in identifier:
            peptides["no_sp"] ##
        elif "|NO_SP|" in identifier:
            peptides["sp"] ##

        print(identifier)
        print(sequence)
        print(metadata)

        
#get_signal_proteins() # list with signal sequences
#get_no_signal_proteins()

main()