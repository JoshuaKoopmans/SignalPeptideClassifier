#open_and_read_file() # return string
#parse_proteins() # dict NO_SP and SP
import domain.Signal_class as sc
def main():
    proteins = open_and_read_file()
    parse_proteins(proteins)
def open_and_read_file():
    with open("../resources/train_set.fasta.txt") as file: #TODO: include file in git
        entries = []
        entry = ""
        for line in file:
            if line.startswith(">"):
                if entry != "":
                    entries.append(entry)
                    entry = ""
            entry += line
        entries.append(entry) #add last entry
    print(entries[0])
    return entries

def parse_proteins(entries):
    peptides = {}
    peptides["no_sp"] = []
    peptides["sp"] = []
    for entry in entries:
        identifier, sequence, metadata = entry.split("\n")[:-1]
        if "|SP|" in identifier:
            peptides["sp"].append(sc.Signal_proteins(identifier,sequence,metadata,True))
        elif "|NO_SP|" in identifier:
            peptides["no_sp"].append(sc.Signal_proteins(identifier,sequence,metadata,False))




main()