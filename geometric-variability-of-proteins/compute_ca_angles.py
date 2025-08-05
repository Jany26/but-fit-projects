import os
import subprocess

import numpy as np


amino_acids_mapping = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL'
}


def compute_ca_skeleton_angles(id: str, chain_dir: str):
    ca_atoms_entries = []
    filename = f"{chain_dir}{id[:4]}_{id[4]}.pdb"
    with open(filename, 'r') as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            # atom_id = line[6:11].strip()
            atom_type = line[13:16].strip()
            # variant = line[16]
            # resname = line[17:20].strip()
            # chain_id = line[21]
            # resid = int(line[22:26].strip())
            x = float(line[31:38].strip())
            y = float(line[39:46].strip())
            z = float(line[47:54].strip())
            if atom_type != "CA":
                continue
            ca_atoms_entries.append(tuple([x,y,z]))
    atoms = np.array(ca_atoms_entries)
    # coords       :  [ Point 0, Point 1, Point 2, Point 3, Point 4 ]
    # coords[:-2]  :  [ Point 0, Point 1, Point 2                   ]
    # coords[1:-1] :  [          Point 1, Point 2, Point 3          ]
    # coords[2:]   :  [                   Point 2, Point 3, Point 4 ]
    vec1 = atoms[:-2] - atoms[1:-1]
    vec2 = atoms[2:] - atoms[1:-1]
    norm_vec1 = np.linalg.norm(vec1, axis=1)
    norm_vec2 = np.linalg.norm(vec2, axis=1)
    vec1_normalized = vec1 / norm_vec1[:, None]
    vec2_normalized = vec2 / norm_vec2[:, None]
    dot_product = np.einsum('ij,ij->i', vec1_normalized, vec2_normalized)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angles = np.arccos(dot_product)
    angles_deg = np.degrees(angles)
    result = np.empty(atoms.shape[0], dtype=object)
    result[0] = np.nan
    result[-1] = np.nan
    result[1:-1] = angles_deg
    return result


def run_tmalign(id1: str, id2: str, chain_dir: str, tm_align_path: str) -> tuple[str, str]:
    filename1 = os.path.join(chain_dir, f"{id1[:4]}_{id1[4]}.pdb")
    filename2 = os.path.join(chain_dir, f"{id2[:4]}_{id2[4]}.pdb")
    result = subprocess.run(
        [tm_align_path, filename1, filename2, "-outfmt", "1"],
        stdout=subprocess.PIPE,
        text=True
    )
    lines = result.stdout.splitlines()
    if len(lines) < 4:
        raise ValueError("Unexpected TMalign output format")
    sequence1 = lines[1].strip()
    sequence2 = lines[3].strip()
    return sequence1, sequence2


def get_residue_mapping(seq1: str, seq2: str, anchor=0, addStr=False) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for i, (res1, res2) in enumerate(zip(seq1, seq2)):
        if res1 != "-":
            if addStr:
                result.append((anchor, amino_acids_mapping[res2]) if res2 != "-" else (None, None))
            else:
                result.append(anchor if res2 != "-" else None)
            if res2 != "-":
                anchor += 1
        if res1 == "-" and res2 != "-":
            anchor += 1
    return result


def get_first_residue_idx(id: str, chain_dir: str, sequence: str) -> list[tuple[int, str]]:
    filename = f"{chain_dir}{id[:4]}_{id[4]}.pdb"
    first_relevant_idx = 0
    anchor: int
    for i, char in enumerate(sequence):
        if char != "-":
            first_relevant_idx = i
            break
    checker = amino_acids_mapping[sequence[first_relevant_idx]]
    with open(filename, 'r') as f2:
        for line in f2:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip()
            resid = line[22:26].strip()
            if resname != checker:
                continue
            anchor = int(resid)
            break
    return anchor


def compute_angles(coords: np.array) -> np.array:
    # coords => np.array (all residues)
    # of np.arrays (for one residue we store 3 atoms -> N, CA, C)
    # of 3d vectors (x y z coordinates)
    vec1 = coords[:, 1] - coords[:, 0]  # CA - N
    vec2 = coords[:, 2] - coords[:, 1]  # C - CA
    dot_product = np.einsum('ij,ij->i', vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1, axis=1)
    norm_vec2 = np.linalg.norm(vec2, axis=1)
    cos_angle = dot_product / (norm_vec1 * norm_vec2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles = np.arccos(cos_angle)
    angles_deg = np.degrees(angles)
    return angles_deg


def clean_coords_by_alignment(
        data: list[list[list]],
        mapping: list[tuple[int, str]]
):
    result = []
    data_idx = 0
    for i, res in mapping:
        if i is None:
            result.append([[None] * 3] * 3)
            continue
        # residue_data = list of three atoms [n_atom_data, ca_atom_data, c_atom_data]
        # atom_data = list containing [resid, resname, atom, x, y, z]

        while (data[data_idx][0][1] != res):
            data_idx += 1
        n_atom, ca_atom, c_atom = None, None, None
        for atom_data in data[data_idx]:
            resid, _, atom, x, y, z, *_ = atom_data
            n_atom = [x, y, z] if atom == "N" else n_atom
            ca_atom = [x, y, z] if atom == "CA" else ca_atom
            c_atom = [x, y, z] if atom == "C" else c_atom
        if None in [n_atom, ca_atom, c_atom]:
            # incomplete data about residues will not contribute to the result
            result.append([[None] * 3] * 3)
        else:
            result.append([n_atom, ca_atom, c_atom])
        data_idx += 1

    return np.array(result, dtype=float)


def compute_angles_all(
    ref_chain_id: str,
    family: dict[str, str],
    chain_dir: str,
    tm_align_path: str,
    debug=False
):
    result = []
    counter = 1
    length = len(family)
    for id, pos in family.items():
        if debug:
            print(f"[info] comparing against {ref_chain_id}: {id} {counter}/{length}")
        seq1, seq2 = run_tmalign(ref_chain_id, id, chain_dir, tm_align_path)
        mapping = get_residue_mapping(seq1, seq2)
        angles = compute_ca_skeleton_angles(id, chain_dir)
        aligned_angles = np.full(len(mapping), np.nan)
        mapping_array = np.array(mapping, dtype=object)
        valid_indices = [i is not None for i in mapping_array]
        valid_mapping = np.array([i for i in mapping_array if i is not None])
        aligned_angles[valid_indices] = angles[valid_mapping]
        result.append(aligned_angles)
        counter += 1
    return np.array(result)


def get_deviations(protein_id, angles: np.array, chain_dir: str, tm_align_path: str):
    _, seq = run_tmalign(protein_id, protein_id, chain_dir, tm_align_path)
    startidx = get_first_residue_idx(protein_id, chain_dir, seq)
    refmap = get_residue_mapping(seq, seq, anchor=startidx, addStr=True)
    std_devs = np.nanstd(angles, axis=0)
    means = np.nanmean(angles, axis=0)
    mean_std = np.nanmean(std_devs)
    std_std = np.nanstd(std_devs)
    z_scores = np.abs((std_devs - mean_std) / std_std)
    categories = np.zeros_like(z_scores, dtype=int)
    categories[(z_scores >= 0) & (z_scores < 0.5)] = 1
    categories[(z_scores >= 0.5) & (z_scores < 1)] = 2
    categories[(z_scores >= 1) & (z_scores < 1.5)] = 3
    categories[(z_scores >= 1.5) & (z_scores < 2)] = 4
    categories[z_scores >= 2] = 5
    result = []

    for i, (res_index, res_name) in enumerate(refmap):
        result.append([res_index, res_name, means[i], std_devs[i], categories[i]])

    return np.array(result, dtype=object)
