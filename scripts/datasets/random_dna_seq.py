import random

def generate_random_fasta(filename, num_sequences=10, seq_length=1000):
    nucleotides = ['A', 'C', 'G', 'T']
    with open(filename, 'w') as f:
        for i in range(num_sequences):
            seq = ''.join(random.choices(nucleotides, k=seq_length))
            f.write(f"{seq}\n")

generate_random_fasta('random_sequences.fasta', num_sequences=13, seq_length=100000)
