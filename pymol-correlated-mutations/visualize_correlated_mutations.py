# xmatuf00 / 222124 / Ján Maťufka / PBI assignment - Visual Correlated Mutation Analysis

import csv
import sys

import pymol


pdb_id = sys.argv[1]
pdb_id = pdb_id.upper()
mi_data_csv = sys.argv[2]

# run pymol in quiet mode and command line mode
pymol.finish_launching()

# clear the screen and set up the 'canvas' for visualization
pymol.cmd.delete('all')
pymol.cmd.fetch(pdb_id)
pymol.cmd.hide("everything")
pymol.cmd.show("wire", pdb_id)
pymol.cmd.color("gray", pdb_id)

# since the residue indexing within the PDB structure is not necessarily the same
# as the indexing within the protein/AA sequence, we compute this mapping
index_to_residue_map = {}
model = pymol.cmd.get_model(pdb_id)
current_index = 0
for atom in model.atom:
    if atom.name == "CA":  # each residue has exactly one alpha carbon
        residue_number = int(atom.resi)
        index_to_residue_map[current_index] = residue_number
        current_index += 1

# Using a PDB structure with `pdb_id`, visualize positions stored in a csv file `top_pairs`
colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink']

# configuring the line design
pymol.cmd.set("dash_width", 1.5)
pymol.cmd.set("dash_gap", 0.0)  # gap of 0 makes it a solid line
with open(mi_data_csv, 'r') as file:
    reader = csv.DictReader(file)
    for idx, row in enumerate(reader):
        res_1_pos = index_to_residue_map[int(row['pdb_prot_pos1'])]
        res_2_pos = index_to_residue_map[int(row['pdb_prot_pos2'])]
        mi = float(row['mutual_information'])
        # we only use 10 colors, but if we wanted to show more pairs, using modulo will make it work
        color = colors[idx % len(colors)]
        linename = f"dist{idx}"
        pymol.cmd.select(f"pair{idx}_1", f"{pdb_id} and resi {res_1_pos}")
        pymol.cmd.select(f"pair{idx}_2", f"{pdb_id} and resi {res_2_pos}")
        pymol.cmd.show("sticks", f"pair{idx}_1")
        pymol.cmd.show("sticks", f"pair{idx}_2")
        pymol.cmd.color(color, f"pair{idx}_1")
        pymol.cmd.color(color, f"pair{idx}_2")
        ca1 = f"pair{idx}_1 and name CA"
        ca2 = f"pair{idx}_2 and name CA"
        dist = float(pymol.cmd.distance(linename, ca1, ca2))

        # making a pseudoatom at the middle of the line for displaying the MI value and distance
        x1, y1, z1 = pymol.cmd.get_atom_coords(ca1)
        x2, y2, z2 = pymol.cmd.get_atom_coords(ca2)
        mid = [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2]
        pymol.cmd.pseudoatom(f"mid{idx}", pos=mid)
        pymol.cmd.label(f"mid{idx}", f'"{dist :.1f} Å, {mi :.2f} MI"')

        # hiding the default value on the distance line so the text does not overlap
        pymol.cmd.color(color, linename)
        # hiding the default value on the distance line so the text does not overlap
        pymol.cmd.hide('labels', linename)

# outputting the image
pymol.cmd.zoom()
pymol.cmd.center()
pymol.cmd.set('antialias', 2)  # smoother
pymol.cmd.set('ray_opaque_background', 1)  # black background
pymol.cmd.bg_color()
pymol.cmd.ray(width=1600, height=1200)
pymol.cmd.png(f'{pdb_id}-CM-pairs.png', dpi=600)

pymol.cmd.do("set keep_alive")  # so that pymol stays on after all is done
