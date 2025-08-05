"""
PROTEIN STRUCTURE

Visualize on the structure of protein X the geometric variation of amino acids
(at corresponding positions) within the protein family to which protein X
belongs in the CATH hierarchy.
You can visualize at your choice e.g. the deviation of the C-alpha angle after
alignment of the structures, or the angles of the side chain with respect
to the nearest neighbours, or any other similar geometric property.
"""

import os
import sys

import pymol

from prepare_data import (
    find_proteins_by_family, get_ref_chains, download_proteins, extract_all_chains, get_cath_db
)
from compute_ca_angles import compute_angles_all, get_deviations, get_residue_mapping


CATH_FILE_PATH = 'cath-b-newest-all'
PROTEIN_ID = sys.argv[1].lower()
PDB_DIR = "./pdbs/"
CHAIN_DIR = "./chains/"
PROGRESS_PRINT = True
TMALIGN_PATH = "TMalign"

color_map = {
    0: "gray",
    1: "green",
    2: "yellow",
    3: "orange",
    4: "salmon",
    5: "red"
}


def prepare_structure_in_pymol(pdb_file: str):
    pymol.finish_launching()
    pymol.cmd.delete("all")
    identifier = pdb_file.split('/')[-1].split('.')[0][0:4]
    identifier += "_protein"
    pymol.cmd.load(pdb_file, identifier)
    pymol.cmd.hide("everything")
    pymol.cmd.set("sphere_scale", 0.4)
    pymol.cmd.color("gray", "all")


def visualize_categories(pdb_file:str, chain_id, data_array):
    identifier = pdb_file.split('/')[-1].split('.')[0][0:4]
    identifier += "_protein"
    for row in data_array:
        residx, resname, angle_mean, angle_dev, category = row
        color = color_map[int(category)]
        selection = f"{identifier} and chain {chain_id} and resi {int(residx)} and name CA"
        pymol.cmd.color(color, selection)


if __name__ == "__main__":
    os.makedirs(PDB_DIR, exist_ok=True)
    os.makedirs(CHAIN_DIR, exist_ok=True)
    get_cath_db(CATH_FILE_PATH)
    result = find_proteins_by_family(PROTEIN_ID, CATH_FILE_PATH)
    # result = {
    #     "1mbnA00": None,
    #     "6tb2A00": None,
    #     "6tb2B00": None,
    #     "6wk3A00": None,
    #     "6wk3B00": None,
    #     "6wk3C00": None,
    #     "6wk3D00": None,
    #     "6zmyA00": None,
    #     "6zmyB00": None
    # }
    ref_chain_ids = get_ref_chains(PROTEIN_ID, CATH_FILE_PATH)
    download_proteins(result, PDB_DIR, debug=PROGRESS_PRINT)
    extract_all_chains(result, PDB_DIR, CHAIN_DIR, debug=PROGRESS_PRINT)
    prepare_structure_in_pymol(f"{PDB_DIR}{PROTEIN_ID}.pdb")
    for chain in ref_chain_ids:
        angles = compute_angles_all(chain, result, CHAIN_DIR, TMALIGN_PATH, debug=PROGRESS_PRINT)
        print(angles)
        print(angles.shape)
        # one row of deviations array: [resi, resn, angle_mean, angle_std, category]
        deviations = get_deviations(chain, angles, CHAIN_DIR, TMALIGN_PATH)
        chain_id = chain[4]
        visualize_categories(f"{PDB_DIR}{PROTEIN_ID}.pdb", chain_id, deviations)

    pymol.cmd.show("spheres", "name ca")
    pymol.cmd.show("sticks", "all")
    pymol.cmd.zoom()
    pymol.cmd.center()
    pymol.cmd.do("set keep_alive")  # so that pymol stays on after all is done
