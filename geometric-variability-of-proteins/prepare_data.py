import subprocess
import os

from Bio.PDB import PDBParser, PDBIO


def get_cath_db(path: str):
    url = "ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/daily-release/newest/cath-b-newest-all.gz"
    archive = "cath-b-newest-all.gz"
    output_path = os.path.join(".", path)
    if os.path.exists(output_path):
        return
    inst = subprocess.run(["wget", url], check=True)
    if inst.returncode != 0:
        raise Exception("[error] couldn't download CATH database")
    subprocess.run(["gunzip", archive], check=True)


def get_ref_chains(pdb_id, cath_db) -> list[str]:
    of = open("get_ref_chains_tempoutfile.txt", 'w')
    subprocess.run(["grep", pdb_id, cath_db], stdout=of)
    of.close()
    result = []
    with open("get_ref_chains_tempoutfile.txt", 'r') as of:
        for line in of:
            words = line.split()
            result.append(words[0])
    os.remove("get_ref_chains_tempoutfile.txt")
    return result


def find_proteins_by_family(pdb_id: str, cath_db: str) -> dict[dict[str, str]]:
    pdb_id = pdb_id.lower()
    superfamily: str = ""
    # map PDB identifier -> sequence position information (start-finish:)
    family_members: dict[str, str] = {}

    with open(cath_db, 'r') as f:
        for line in f:
            if line.startswith(pdb_id):
                _, _, superfamily, _ = line.split()
                break

        f.seek(0)

        for line in f:
            id, _, family, pos = line.split()
            if family == superfamily:
                family_members[id] = pos
    return family_members


def prune_family(family: dict[str, str], failset: set[str], debug=False):
    keys_to_delete = [
        key for key in family if any(key.startswith(pdb) for pdb in failset)
    ]
    for key in keys_to_delete:
        del family[key]


def download_proteins(mapping: dict[str, str], pdb_dir: str, debug=False):
    present_pdbs = {i.split('.')[0] for i in os.listdir(pdb_dir)}
    set_failed = set()
    if debug:
        print(f"[info] there are {len(present_pdbs)} present proteins found within the specified folder")
    pdbs: set[str] = {id[0:4] for id, pos in mapping.items()}
    if debug:
        print(f"[info] trying to download {len(pdbs)} unique pdb files")
    sorted_pdbs = list(pdbs)
    sorted_pdbs.sort()
    length = len(sorted_pdbs)
    downloaded = 0
    skipped = 0
    failed = 0
    for i, id in enumerate(sorted_pdbs):
        if debug:
            print(f"[info] downloading protein {id}: {i}/{length}")
        ret = download_pdb_with_wget(id, output_dir=pdb_dir, debug=debug)
        downloaded += 1 if ret is not None and ret != "" else 0
        skipped += 1 if ret is not None and ret == "" else 0
        failed += 1 if ret is None else 0
        if ret is None:
            set_failed.add(id)
    if debug:
        print(f"[info] protein download finished: downloaded {downloaded}, skipped {skipped}, failed {failed}")
    prune_family(mapping, set_failed)


def download_pdb_with_wget(pdb_id: str, output_dir=".", debug=False):
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")

    if os.path.exists(output_path):
        return ""
    try:
        subprocess.run(["wget", "-q", "-O", output_path, url], check=True)
    except subprocess.CalledProcessError as e:
        if debug:
            print(f"[warn] couldn't download {pdb_id}")
        return None
    
    return output_path


def extract_all_chains(family: dict[str, str], pdb_dir: str, chain_dir: str, debug=False):
    if debug:
        print(f"[info] extracting {len(family)} chains")
    extracted = 0
    skipped = 0
    failed = 0
    set_failed = set()
    for id, pos in family.items():
        ret = extract_chain(id, pdb_dir, output_dir=chain_dir)
        extracted += 1 if ret is not None and ret != "" else 0
        skipped += 1 if ret is not None and ret == "" else 0
        failed += 1 if ret is None else 0
        if ret is None:
            set_failed.add(id)
        if ret is None:
            print(f"[warn] failed to extract chain {id}")
    if debug:
        print(f"[info] chain extraction done: extracted {extracted}, skipped {skipped}, failed {failed}")
    prune_family(family, set_failed)


def extract_chain(cath_id, pdb_dir, output_dir):
    pdb_id = cath_id[:4]
    chain_id = cath_id[4]
    pdb_file = f"{pdb_dir}{pdb_id}.pdb"
    chain_file = f"{pdb_id}_{chain_id}.pdb"
    output_path = os.path.join(output_dir, f"{chain_file}")
    if os.path.exists(output_path):
        return ""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # since there are multiple conformations/models of a protein,
    # we try to consider the most-relevant (usually the first model)
    # for first_model in structure:

    for chain in structure[0]:
        if chain.get_id() == chain_id:
            io = PDBIO()
            io.set_structure(chain)
            io.save(output_path)
            return output_path
    return None

