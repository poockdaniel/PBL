import csv

# Converts a tsv file to fasta format, including EC Numbers
# Copied and adapted from CLEAN (src/CLEAN/utils.py)
def csv_to_fasta(csv_name, fasta_name):
    csvfile = open(csv_name, 'r')
    csvreader = csv.reader(csvfile, delimiter='\t')
    outfile = open(fasta_name, 'w')
    for i, rows in enumerate(csvreader):
        if i > 0:
            outfile.write('>sp|' + rows[0] + '|' + rows[1] + ' EC=' + rows[2] + '\n')
            outfile.write(rows[3] + '\n')