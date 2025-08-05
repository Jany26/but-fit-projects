# BIF project - Prediction of Impacts of Mutations on Protein Function
# Ján Maťufka / xmatuf00 / 222124
# <xmatuf00@stud.fit.vutbr.cz>
# 14.05.2024


# Note: For some unknown reason, the results might be off by ~0.01.
# Normalized properties, and leaf distances are correct.
# Score computation in task 3 introduces some floating point errors.
# Perhaps the order of operations is incorrect.
# Based on result values comparison with other students.


import sys
import newick  # developed using newick version 1.9.0 (Python 3.10.12)

amino_acids_ordered = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

# controls whether the following 5 (subtask result) files are created
subresults = False

task1_output_file = "task1-absolute-distances.csv"
task2_output_file = "task2-normalized-distances.csv"
task3a_output_file = "task3-sequences.csv"
task3_output_file = "task3-score.csv"
task4_output_file = "task4-properties.csv"

task5_output_file = "result.csv"  # change filename if needed, None -> stdout

# Subresult printing to csv
def csv_printout(filename: str, headers: list[str], content: dict, output: bool):
    if output:
        with open(filename, "w") as f:
            f.write(f"{','.join(h for h in headers)}\n")
            for key, value in content.items():
                f.write(f"{key},{value}\n")


# Task 1: Obtaining leaf distances from the roof.
# BFS-like traversal of the phylogenetic tree, while calculating
# leaf-path distances from the parsed Newick format
def get_distances(tree_file_path: str, output=False) -> dict[str, float]:
    tree = newick.read(tree_file_path)[0]  # 1 tree in the forest expected
    result: dict[str, float] = {}

    # bfs-like traversal while computing distances
    queue = [(tree, tree.length)]
    while queue != []:
        current, length = queue.pop(0)
        if current.descendants == []:  # detecting leaves
            result[current.name] = length
        for i in current.descendants:
            queue.append((i, length + i.length))

    global task1_output_file
    csv_printout(task1_output_file, ["alignment", "leaf_distance"], result, output)
    return result


# Task 2: Min-max normalization of leaf distances.
def normalize_distances(distances: dict[str, float], output=False) -> dict[str, float]:
    minval, maxval = min(distances.values()), max(distances.values())
    norm = {name: (val - minval) / (maxval - minval) for name, val in distances.items()}

    global task2_output_file
    csv_printout(
        task2_output_file, ["alignment", "normalized_leaf_distance"], norm, output
    )
    return norm


# Parse the FASTA file with multi-sequence alignment.
def load_aligned_sequences(file_path: str, output=False):
    result = {}
    sequence = ""
    name = None
    for line in open(file_path, "r"):
        if not line.startswith(">"):
            sequence += line.strip()
            continue
        if name is not None:
            result[name] = sequence
        name = line.lstrip(">").rstrip()
        sequence = ""

    global task3a_output_file
    csv_printout(task3a_output_file, ["name", "sequence"], result, output)
    return result


# Task 3: Compute the conservation score for every aminoacid.
def get_score(
    weights: dict[str, float], sequences: dict[str, str], output=False
) -> dict[str, list[float]]:
    length = max(len(seq) for seq in sequences.values())
    score_matrix = {i: [0.0] * length for i in amino_acids_ordered}
    score_matrix["-"] = [0.0] * length

    weight_sum = sum(weights.values())

    # we go through all sequences and all columns and add corresponding
    for name, seq in sequences.items():
        for col, aa in enumerate(seq):
            # and add corresponding sequence weight/distance
            score_matrix[aa][col] += weights[name]
    # after computing sums for each cell of the matrix,
    # we divide the values by the sum of sequence weights
    for aa in amino_acids_ordered:
        for col in range(length):
            score_matrix[aa][col] /= weight_sum
    del score_matrix["-"]

    global task3_output_file
    csv_printout_score_matrix(task3_output_file, score_matrix, output)

    return score_matrix


# CSV-like printout of the score matrix (for subresult checking).
def csv_printout_score_matrix(
    filename: str, matrix: dict[str, list[float]], output: bool
):
    if not output:
        return
    length = 0
    for j in matrix.values():
        length = len(j)
        break
    with open(filename, "w") as f:
        f.write("AA," + ",".join([str(i + 1) for i in range(length)]) + "\n")
        for aa, row in matrix.items():
            f.write(f"{aa}," + ",".join([str(v) for v in row]) + "\n")


# Helper function: Check if list of strings is convertible to list of floats
# Returns list of floats if all are convertible, otherwise None.
def all_floats(items):
    try:
        result = [float(x) for x in items]
        return result
    except ValueError:
        return None


# Task 4: Calculate the normalized properties for every aminoacid.
# Returns a matrix of normalized values of physical/chemical properties
# of aminoacids (each row = 1 aminoacid, each column = different property)
def get_physical_properties(file_path: str, output=False) -> list[list[float]]:
    result = []
    names = []  # for csv output
    property_values = []  # for temporarily storing values of 1 property
    name_row = True  # for filtering out rows that contain AA names
    for line in open(file_path, "r"):
        words = [s for s in line.strip().split() if s]
        converted = all_floats(words)
        if converted is not None:  # we have row of floats
            property_values.extend(converted)
            name_row = True  # after extracting all consecutive rows with
            continue  # values, a row with property name should follow
        if name_row:
            names.append(" ".join(words))
            name_row = False
        if property_values != []:
            result.append(property_values)
            property_values = []
    result.append(property_values)

    normalized = []
    for vals in result:
        minval, maxval = min(vals), max(vals)
        normalized.append([(i - minval) / (maxval - minval) for i in vals])

    transpose = [list(row) for row in zip(*normalized)]
    if output:
        global task4_output_file
        with open(task4_output_file, "w") as f:
            f.write("AA," + ",".join(names) + "\n")
            for i, vals in enumerate(transpose):
                aa = amino_acids_ordered[i]
                f.write(f"{aa}," + ",".join([str(v) for v in vals]) + "\n")
    return transpose


# Task 5: Final computation of mutation scores.
# Sum of differences between products of each property with the given
# sequence weight (query / others).
def difference_scores(
    query_seq: str,
    score: list[list[float]],
    properties: list[list[float]],
    output_file=None,
):
    # compute difference score
    output_matrix = [[None] * len(query_seq) for _ in amino_acids_ordered]
    property_count = len(properties[0])
    for col, symbol in enumerate(query_seq):
        if symbol == "-":
            continue
        idx = amino_acids_ordered.index(symbol)
        query_w = score[symbol][col]
        for row, aa in enumerate(amino_acids_ordered):
            comp_w = score[aa][col]
            diff_score = sum(
                query_w * properties[idx][i] - comp_w * properties[row][i]
                for i in range(property_count)
            )
            output_matrix[row][col] = diff_score

    # if file specified, print to file, otherwise stdout
    out = sys.stdout if output_file is None else open(output_file, "w")

    # print header
    header = "AA," + ",".join([str(i + 1) for i in range(len(query_seq))])
    print(header, file=out)

    # write them to output by rows
    for i, row in enumerate(output_matrix):
        aa = amino_acids_ordered[i]
        print(
            f"{aa}," + ",".join(("-" if i is None else str(i)) for i in row), file=out
        )


# script expects 3 arguments:
#   - 1: path to tree file in the newick format [tree.tre]
#   - 2: path to file with sequence alignment (FASTA) [msa.fasta]
#   - 3: path to file with physical properties of aminoacids [aaindex.txt]
# final results are printed to "result.csv" by default
if __name__ == "__main__":
    if len(sys.argv) >= 4:
        distances = get_distances(sys.argv[1], output=subresults)
        norm_distances = normalize_distances(distances, output=subresults)
        sequences = load_aligned_sequences(sys.argv[2], output=subresults)
        score = get_score(norm_distances, sequences, output=subresults)
        props = get_physical_properties(sys.argv[3], output=subresults)
        output = difference_scores(sequences["query"], score, props, task5_output_file)
    else:
        print(f"not enough argumnets")
        print(f"usage: python3 bif.py tree-file sequence-file properties-file")
