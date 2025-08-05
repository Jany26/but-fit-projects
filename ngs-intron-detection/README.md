# Next-gen Sequencing - práca so sekvenačnými dátami

kurz: PBI - Pokročilá Bioinformatika
meno: Ján Maťufka  
mail: xmatuf00@stud.fit.vutbr.cz  
VUT ID: 222124  

## Obsah archívu:
`get_gene_expression.py` - skript na výpočet génovej expresie
`table.csv` - výsledná expresia jednotlivých génov zo zhluku S100 v sekvenačných dátach experimentu SRR1777101
`intron_proof.png` - screenshot IGV poukazujúci na existenciu intrónov v rámci génu S100A10
`README.md` - tento postup

## Ďalšie potrebné súbory pre reprodukciu výsledkov

Keďže tieto dáta majú obrovský objem, neboli pridané do archívu. Spôsob, ako sa k ním dopracovať:

1) Nasekvenovaný ľudský genóm
- `Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz`
- 942.5 MB
```sh
wget http://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz
gunzip -k Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz
```

2) Anotácia genómu
- `Homo_sapiens.GRCh38.109.chr.gtf.gz`
- 54.3 MB
```sh
wget ftp://ftp.ensembl.org/pub/release-109/gtf/homo_sapiens/Homo_sapiens.GRCh38.109.chr.gtf.gz
gunzip -k Homo_sapiens.GRCh38.109.chr.gtf.gz
```


3) Experimentálne dáta
- `SRR1777101.fastq.gz`
- 285.4 MB
- z linku [Trace.NCBI](https://trace.ncbi.nlm.nih.gov/Traces/index.html?view=run_browser&acc=SRR1777101&display=download) pod tabom FASTA/FASTQ Download -> tlačítko FASTQ
```sh
gunzip -k SRR1777101.fastq.gz
```

## Zistenie génovej expresie

```sh
hisat2-build Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa hg_index
hisat2 -x hg_index -U SRR1777101.fastq -S aligned_reads.sam --rna-strandness R
samtools view -bS aligned_reads.sam > aligned_reads.bam
samtools sort aligned_reads.bam -o aligned_reads_sorted.bam
samtools index aligned_reads_sorted.bam
featureCounts -a Homo_sapiens.GRCh38.109.chr.gtf -o gene_counts.tsv aligned_reads_sorted.bam
python3 get_gene_expression.py >output.csv
```

Súbor `gene_counts.txt` vo formáte TSV obsahuje pre nás relevantné informácie o danom experimente:
- všetky gény ktoré sa dali nájsť v sekvenovaných dátach
- ku každému génu je navyše informácia:
    - na ktorom chromozóme sa gén nachádza, z ktorého vlákna pochádza,
    - počiatočné a koncové indexy jednotlivých exónov,
    - počet readov ktoré sa namapovali do kódovacej oblasti génu,
    - dĺžka génu.

Python skript `get_gene_expression.py` spracuje tento súbor a vyfiltruje nám informácie o génoch zo zhluku S100 (prekladom indexov v tvare `ENSG000...` do názvov génov pomocou súboru `Homo_sapiens.GRCh38.109.chr.gtf`), a vypočíta génovú expresiu v jednotkách RPKM, TPM.

Z výslednej tabuľky je vidieť, že len 4 gény zo zhluku S100 (S100A10, S100A16, S100A14, S100A13) majú nenulovú expresiu.
Pre vizualizáciu intrónov bol vybraný gén S100A10, keďže má najväčšiu expresiu.

## Odhalenie prítomnosti intrónov

Pre správne zobrazenie RNA dát v IGV bolo potrebné najprv zoradiť súbor s anotáciami génov.
```sh
igvtools index Homo_sapiens.GRCh38.109.chr.gtf
sort -k1,1 -k4,4n Homo_sapiens.GRCh38.109.chr.gtf > Homo_sapiens.GRCh38.109.chr.sorted.gtf
bgzip Homo_sapiens.GRCh38.109.chr.sorted.gtf
tabix -p gff Homo_sapiens.GRCh38.109.chr.sorted.gtf.gz
```

Po spustení IGV je potrebné doň načítať nasledovné súbory (vľavo hore File - Load from File...):
- `aligned_reads_sorted.bam`
- `aligned_reads_sorted.bam.bai`
- `Homo_sapiens.GRCh38.109.chr.sorted.gtf.gz.tbi`

Po načítaní možno do políčok pod horným menu vybrať:
- `Human (GRCh38/hg38) | chr1 | chr1:151,982,915-151,993,859`

Použité indexy možno vyčítať zo súboru `gene_counts.tsv` (nájdením riadka s ID génu ENSG00000197747, zaujíma nás štartovací index prvého exónu a koncový index posledného exónu).

Po stlačení "Go" sa IGV nastaví presne tak, ako je na priloženom obrázku, červeno-modré oblúky v géne S100A10 sú intróny, stĺpce sivých obdĺžnikov sú ready, ktoré sa namapovali na kódujúce oblasti (exóny) nášho génu.

