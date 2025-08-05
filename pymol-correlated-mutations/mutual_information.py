# xmatuf00 / 222124 / Ján Maťufka / PBI assignment - Visual Correlated Mutation Analysis

import sys
from collections import Counter

import numpy as np
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment as MSA

alignment: MSA = AlignIO.read(sys.argv[1], "fasta")
sequences = {str(record.id): str(record.seq) for record in alignment}
# transpose the alignment, so when we read row by row, we basically iterate over columns of the MSA
alignment_cols = np.array([list(seq) for seq in sequences.values()]).T

def calculate_mi(col1, col2):
    joint_prob = Counter(zip(col1, col2))  # counts of tuples
    total_pairs = sum(joint_prob.values())  # number of tuples
    
    p1 = Counter(col1)
    p2 = Counter(col2)
    
    mi = 0
    for (a, b), joint_count in joint_prob.items():
        p_joint = joint_count / total_pairs
        p_a = p1[a] / total_pairs
        p_b = p2[b] / total_pairs
        mi += p_joint * np.log2(p_joint / (p_a * p_b))
    
    return mi

num_columns = alignment_cols.shape[0]

# if the key (position in the MSA) is not in the mapper,
# that position is a gap and thus does not map to a specific position in the PDB protein sequence
no_gap_seq_mapper: dict[int, int] = {}
counter = 0  # we only count positions that do not contain '-' in the reference protein 
ref_protein_seq = sequences[sys.argv[2]]

# here we store the correlated positions and their MI values
all_results: list[tuple[int, int, float]] = []
for i in range(num_columns):
    # we start from i+1, since we do not care about values on the matrix diagonal
    if ref_protein_seq[i] != '-':
        no_gap_seq_mapper[i] = counter
        counter += 1
    for j in range(i + 1, num_columns):
        # we do not need to compute the [j, i] value since the MI matrix is symmetric
        all_results.append(tuple([i, j, calculate_mi(alignment_cols[i], alignment_cols[j])]))

# sort the correlated positions by the mutual information, descending
sorted_results = sorted(all_results, key=lambda x: x[2], reverse=True)

header = ['msa_pos1', 'msa_pos2', 'pdb_prot_pos1', 'pdb_prot_pos2', 'mutual_information']
print(",".join(header))
counter = 0
for pos1, pos2, res in sorted_results:
    # we only want to show the top 10 strongest correlated pairs
    if counter >= 10:
        break
    # the correlated positions actually have to be in our reference protein from PDB
    if not (pos1 in no_gap_seq_mapper and pos2 in no_gap_seq_mapper):
        continue
    # obviously we do not want the pairs to be the same position
    if pos1 == pos2:
        continue
    print(f"{pos1},{pos2},{no_gap_seq_mapper[pos1]},{no_gap_seq_mapper[pos2]},{res}")
    counter += 1
