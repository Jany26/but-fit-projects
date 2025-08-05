NGS EXERCISE:

1. Find gene cluster S100A in human genome.
    - use this context: chr1:153,300,000-153,700,000
    - relevant sequences within the cluster: S100A9, S100A12, S100A8, S100A7a, S100A7L2, S100A7, S100A6, S100A5, S100A4, S100A3, S100A2, S100A16, S100A14, S100A13, S100A1

COLLECTING BASIC DATA AND INFORMATION

2. Using "Genome Browser", "Table Browser", or some additional processing, get:
    a) AA sequence of the relevant part (for exercise purposes) and relevant (whole) chromosome (for mapping purposes) in FASTA format.
        - using View - DNA in the menu at the top
    b) Borders/Boundaries of the listed genes, getting mainly transcripts is enough, in GFF3 format (including exons).
        - using Table Browser - position - get GTF (or alternatively, BED) - convert to GFF3

3. Create a table for machine translation of UCSC transcript names (for example "uc001fbq.3") to gene names (S100A9)
    - Table Browser - table kgAlias -> filter names using "S100A" as a prefix

(note: steps 1, 2 and 3 are not required for the rest of the tasks)

4. Using DNA-Seq data, identify SNP (single nucleotide polymorphism) in coding sequences of individual transcripts
    - mapping to the reference genome (chromosome) using BWA or Bowtie2
    - using samtools to convert SAM to BAM
    - creating VCF file using samtools pileup
    - inspection in IGV or other browser

5. Display annotations and mapped sequences in IGV

6. Filter out reads from step 5., which do not map into the S100A cluster, and try to construct examined region with the rest, using Velvet.

ASSESSMENT PART:

Note: This process is similar to steps 4-6, but instead of DNA we analyze RNA.

7. Using RNA-Seq data from the link below, estimate expression of individual members of the S100A cluster in an experiment of your choosing,
( compute reads that map into specific areas, see https://www.biostars.org/p/11105/ ).

- During mapping, take into account the existence of introns (usually missing in RNA-Seq) and find a concrete proof of intron's presence in the transcripts or an alternative splicing (e.g. use tophat/hisat2 instead of bwa/bowtie).
- Submit:
    - a reproducible procedure (a bash script or a readme with commands to run, or similar) with commentary
    - a table with detected normalized expression (RPKM / TPKM) of genes-of-interest (those that are in the cluster's region and have S100 in their name)
    - a visualisation of a detected intron's example in IGV or another suitable genome-browser

PROVIDED LINKS:
DNA-Seq and RNA-Seq data for the task: http://trace.ncbi.nlm.nih.gov/Traces/study/?acc=SRP052901
    - use just one "run" (SRRxxxxxx)
DNA-Seq data: https://www.ncbi.nlm.nih.gov/sra/SRX1620434[accn] (or a subset of it)

PROVIDED FILES:
(not sure if all are relevant for the assessment part)
- `forward.fq.gz` (probably contains the whole chromosome1 DNA sequence)
- `reverse.fq.gz` (probably contains the reverse strain of the whole chromosome1 DNA sequence)
- `S100_cluster_mapped.fastq` (relevant part of the genome - S100 gene cluster)

- - - - - - - - - - - - - - - - - - - - - - - - -

Unsure what this is for, but it was included:
From: https://ucdavis-bioinformatics-training.github.io/2019_March_UCSF_mRNAseq_Workshop/data_reduction/alignment.html
Many alignment algorithms to choose from.

Spliced Aligners
  STAR
  HiSAT2 (formerly Tophat [Bowtie2])
  GMAP - GSNAP
  SOAPsplice
  MapSplice

Aligners that can ’clip’
  bwa-mem
  Bowtie2 in local mode

