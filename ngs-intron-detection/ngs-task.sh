# download SRR1777101.fastq.gz from:
# https://trace.ncbi.nlm.nih.gov/Traces/index.html?view=run_browser&acc=SRR1777101&display=download
# under FASTA/FASTQ Download tab -> FASTQ button
# unzip with 

# download
hisat2-build Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa hg_index
hisat2 -x hg_index -U SRR1777101.fastq -S aligned_reads.sam --rna-strandness R
samtools view -bS aligned_reads.sam > aligned_reads.bam
samtools sort aligned_reads.bam -o aligned_reads_sorted.bam
samtools index aligned_reads_sorted.bam

# GTF
wget ftp://ftp.ensembl.org/pub/release-109/gtf/homo_sapiens/Homo_sapiens.GRCh38.109.chr.gtf.gz
gunzip Homo_sapiens.GRCh38.109.chr.gtf.gz

featureCounts -a Homo_sapiens.GRCh38.109.chr.gtf -o gene_counts.txt aligned_reads_sorted.bam

python3 get_rpkm.py