# xmatuf00 / Ján Maťufka / 222124

import pandas as pd
import re


GTF_FILE = "Homo_sapiens.GRCh38.109.chr.gtf"
GENE_COUNTS_SUMMARY_FILE = "gene_counts.tsv"
SUBSTRING_OF_INTEREST = "S100"


def get_rpkm_tpm_from(summary_file: str) -> pd.DataFrame:
    data = pd.read_csv(summary_file, sep="\t", comment="#")
    data.columns = ["Geneid", "Chr", "Start", "End", "Strand", "Length", "Counts"]

    # Compute RPKM (reads per kilobase of transcript per million reads mapped)
    # according to https://www.metagenomics.wiki/pdf/qc/RPKM

    # RPKM =  numReads          / (geneLength / 1 000 * totalNumReads / 1 000 000)
    # RPKM =  numReads          / (geneLength * 10^-3 * totalNumReads * 10^-6)
    # RPKM = (numReads * 10^9)  / (geneLength *         totalNumReads)

    total_mapped_reads = data["Counts"].sum()
    data["RPKM"] = (data["Counts"] * 1e9) / (data["Length"] * total_mapped_reads)

    # Compute (gene-level) TPM (Transcripts per million)
    # 
    # It is possible to compute exon-level TPM, but for that we would need to have
    # information about how many reads map into individual exons, not only to whole
    # genes.
    rpkm_sum = data["RPKM"].sum()
    data["TPM"] = (data["RPKM"] / rpkm_sum) * 1e6
    data[["Geneid", "RPKM", "TPM"]].to_csv("normalized_gene_expression.csv", index=False)

    return data[["Geneid", "RPKM", "TPM"]]


def get_gene_name_to_id_conversion(gtf_file: str) -> dict[str, str]:
    gene_mapping = {}
    with open(gtf_file, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue
            columns = line.strip().split('\t') 
            if columns[2] != "gene":
                continue
            attributes = columns[8]
            gene_id_match = re.search(r'gene_id "([^"]+)"', attributes)
            gene_name_match = re.search(r'gene_name "([^"]+)"', attributes)
            if not (gene_id_match and gene_name_match):
                continue
            gene_id = gene_id_match.group(1)
            gene_name = gene_name_match.group(1)
            gene_mapping[gene_id] = gene_name
    return gene_mapping


if __name__ == "__main__":
    translation = get_gene_name_to_id_conversion(GTF_FILE)
    expression_table = get_rpkm_tpm_from(GENE_COUNTS_SUMMARY_FILE)
    # Header print
    print(f"{'gene_name':15}, {'gene_id':20}, {'rpkm':10}, {'tpm':10}")
    for gid, gname in translation.items():
        if SUBSTRING_OF_INTEREST in gname:
            rpkm = expression_table.loc[expression_table['Geneid'] == gid, 'RPKM'].values[0]
            tpm = expression_table.loc[expression_table['Geneid'] == gid, 'TPM'].values[0]
            print(f"{gname:15}, {gid:20}, {rpkm:<10.5f}, {tpm:<10.5f}")

