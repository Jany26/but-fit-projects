# Simulovaná metagenomická analýza

kurz: PBI - Pokročilá Bioinformatika  
meno: Ján Maťufka  
mail: xmatuf00@stud.fit.vutbr.cz  
VUT ID: 222124  

## Obsah archívu:
`README.md` - tento súbor, popisuje postup práce, a ďalšie potrebné informácie pre reprodukciu výsledkov
`megan-visual.png` - screenshot z MEGAN s priblížením uzlov popisujúcich naše cieľové organizmy
`megan-visual-clear.png` - screenshot z MEGAN kde je vidieť celý fylogenetický strom (bez popisov nám nezaujímavých organizmov)
`megan-visual-full.png` - schreenshot z MEGAN kde vidieť celý fylogenetický strom (po úroveň záujmu)
`metagenome.megan` - výstup z MEGAN, ktorý možno do programu načítať a tak prezrieť celý strom
`contigs.fa` - výstup prvej fázy projektu (contigy poskladané nástrojom VELVET)
`diamond-output-sample.m8` - vzorka výstupu z programu DIAMOND

Ostatné súbory (medzivýstupy analýzy) budú dostupné na tomto odkaze na GDrive:
https://drive.google.com/drive/folders/1YDV1b2_b46YX2haSzYfPjzzm5QGTBPzh?usp=sharing


Adresár na GDrive obsahuje:
- plný výstup z programu DIAMOND (.m8) - 320 MB
- plný výstup binningu z programu MEGAN (.rma6) - 113 MB

## Vzorkovacie pomery

Keďže našou úlohou je nejak presne namiešať dáta z daných experimentov, potrebujeme vedieť, akým spôsobom ich navzorkovať.
Genómy baktérií E.coli a Salmonella majú dĺžku 5-6 miliónov bází, avšak dáta zo sekvenačných readov obsahujú aj niekoľko sto miliónov.
Preto je dôležité vhodne zvoliť vzorkovacie pomery pre jednotlivé dáta.

Na to sme zvolili prístup vybrať 20 rôznych `strain` (variácií) daných druhov baktérií z NCBI databázy, a vypočítať priemernú dĺžku genómov.

| identifier   | organism            | strain                | genome size | date      |
| ------------ | ------------------- | --------------------- | ----------- | --------- |
| ASM2436490v1 | Salmonella enterica | SC2016090 (strain)    | 5.248 Mb    | Jul, 2022 |
| ASM2436486v1 | Salmonella enterica | SC2016025 (strain)    | 5.245 Mb    | Jul, 2022 |
| ASM332503v1  | Salmonella enterica | SA20100201 (strain)   | 5.195 Mb    | Jul, 2018 |
| ASM972991v1  | Salmonella enterica | FDAARGOS_708 (strain) | 5.137 Mb    | Dec, 2019 |
| ASM183155v2  | Salmonella enterica | CFSAN044865 (strain)  | 5.084 Mb    | Jun, 2021 |
| ASM1933942v1 | Salmonella enterica | LHICA_E3 (strain)     | 5.082 Mb    | Jul, 2021 |
| ASM2253354v1 | Salmonella enterica | P164045 (strain)      | 5.314 Mb    | Mar, 2022 |
| ASM332513v1  | Salmonella enterica | SA20083530 (strain)   | 5.263 Mb    | Jul, 2018 |
| ASM2073634v1 | Salmonella enterica | SZL 38 (strain)       | 5.132 Mb    | Nov, 2021 |
| ASM696924v1  | Salmonella enterica | MAC15 (strain)        | 5.052 Mb    | Jul, 2019 |
| ASM2953731v1 | Salmonella enterica | SC2020597 (strain)    | 5.090 Mb    | Apr, 2023 |
| ASM4192729v1 | Salmonella enterica | SA165 (strain)        | 5.042 Mb    | Sep, 2024 |
| ASM1846004v1 | Salmonella enterica | no75 (strain)         | 5.283 Mb    | May, 2021 |
| ASM4193034v1 | Salmonella enterica | SA40 (strain)         | 5.031 Mb    | Sep, 2024 |
| ASM2333084v1 | Salmonella enterica | 143 (strain)          | 5.390 Mb    | May, 2022 |
| ASM2287068v1 | Salmonella enterica | 1722 (strain)         | 5.239 Mb    | Apr, 2022 |
| ASM2333072v1 | Salmonella enterica | 1559 (strain)         | 5.297 Mb    | May, 2022 |
| ASM2147424v1 | Salmonella enterica | 2017028-SE (strain)   | 5.271 Mb    | Jan, 2022 |
| ASM2147428v1 | Salmonella enterica | 2017005-SE (strain)   | 5.312 Mb    | Jan, 2022 |
| ASM4527820v1 | Salmonella enterica | SA746 (strain)        | 5.187 Mb    | Nov, 2024 |
| ASM2792582v1 | Escherichia coli    | 10153 (strain)        | 6.108 Mb    | Jan, 2023 |
| ASM4057135v1 | Escherichia coli    | 12089 (strain)        | 5.893 Mb    | Jul, 2024 |
| ASM301803v1  | Escherichia coli    | 2015C-4944 (strain)   | 5.901 Mb    | Mar, 2018 |
| ASM2792574v1 | Escherichia coli    | EH2252 (strain)       | 6.114 Mb    | Jan, 2023 |
| ASM2792580v1 | Escherichia coli    | PV0838 (strain)       | 5.848 Mb    | Jan, 2023 |
| ASM2792576v1 | Escherichia coli    | 98E11 (strain)        | 5.965 Mb    | Jan, 2023 |
| ASM4432487v1 | Escherichia coli    | GF60 (strain)         | 6.140 Mb    | Nov, 2024 |
| ASM2792578v1 | Escherichia coli    | NIID080884 (strain)   | 5.820 Mb    | Jan, 2023 |
| ASM2792556v1 | Escherichia coli    | EH031 (strain)        | 5.922 Mb    | Jan, 2023 |
| ASM2792584v1 | Escherichia coli    | 02E060 (strain)       | 5.883 Mb    | Jan, 2023 |
| ASM3238231v1 | Escherichia coli    | TUM9803 (strain)      | 6.025 Mb    | Oct, 2023 |
| ASM1405844v2 | Escherichia coli    | MBT-5 (strain)        | 5.936 Mb    | Sep, 2020 |
| ASM172112v1  | Escherichia coli    | FORC_028 (strain)     | 5.704 Mb    | Sep, 2016 |
| ASM2792550v1 | Escherichia coli    | 2313 (strain)         | 5.906 Mb    | Jan, 2023 |
| ASM435840v1  | Escherichia coli    | CFSAN027343 (strain)  | 5.778 Mb    | Mar, 2019 |
| ASM4057136v1 | Escherichia coli    | 12867 (strain)        | 5.693 Mb    | Jul, 2024 |
| ASM301857v1  | Escherichia coli    | 2013C-4538 (strain)   | 5.769 Mb    | Mar, 2018 |
| ASM301845v1  | Escherichia coli    | 97-3250 (strain)      | 6.156 Mb    | Mar, 2018 |
| ASM2430068v1 | Escherichia coli    | 2003-3014 (strain)    | 5.955 Mb    | Jul, 2022 |
| ASM396646v1  | Escherichia coli    | E2865 (strain)        | 6.008 Mb    | Jul, 2018 |

Salmonella enterica - avg genome size       = 5.1947 Mb
Escherichia coli    - avg genome size       = 5.9262 Mb

`coverage          =   # of bases in reads / # of bases in genome`
`subsampling ratio =   desired coverage    / actual coverage`

Tu sú medzivýsledky použité k výpočtu vzorkovacích pomerov pre jednotlivé druhy:

| taxon               | genome    | reads    | coverage | subsampling ratio |
| ------------------- | --------- | -------- | -------- | ----------------- |
| Salmonella enterica | 5.1947 Mb | 377.2 Mb | 72.6125x | 0.16526           |
| Escherichia coli    | 5.9262 Mb | 438.6 Mb | 74.0103x | 0.05405           |

0.16526 = 12 / 72.6125
0.05405 = 4 / 74.0103

## Príprava dát (vzorkovanie, miešanie)

Podľa identifikátorov v zadaní sme si získali dáta z NCBI databázy:

```sh
fastq-dump --split-files --gzip SRR26893485
fastq-dump --split-files --gzip SRR26893959
```

Na základe vypočítaných vzorkovacích pomerov sme teda nasledovným spôsobom pripravili párové ready
(pre náhodné vzorkovanie bol použitý seed 42):

```sh
seqkit pair -1 SRR26893485_1.fastq.gz -2 SRR26893485_2.fastq.gz -u
seqkit pair -1 SRR26893959_1.fastq.gz -2 SRR26893959_2.fastq.gz -u
seqkit sample -p 0.05405 -s 42 SRR26893485_1.paired.fastq.gz -o ecoli_1.fastq.gz
seqkit sample -p 0.05405 -s 42 SRR26893485_2.paired.fastq.gz -o ecoli_2.fastq.gz
seqkit sample -p 0.16526 -s 42 SRR26893959_1.paired.fastq.gz -o salmonella_1.fastq.gz
seqkit sample -p 0.16526 -s 42 SRR26893959_2.paired.fastq.gz -o salmonella_2.fastq.gz
cat ecoli_1.fastq.gz salmonella_1.fastq.gz > concat_1.fastq.gz
cat ecoli_2.fastq.gz salmonella_2.fastq.gz > concat_2.fastq.gz
seqkit shuffle concat_1.fastq.gz -o mixed_1.fastq.gz
seqkit shuffle concat_2.fastq.gz -o mixed_2.fastq.gz
seqkit pair -1 mixed_1.fastq.gz -2 mixed_2.fastq.gz -u
```

Tu je tabuľka obsahujúca výstupy z príkazu `seqkit stats ` nad použitými súbormi:

| file                   | format | type | num_seqs |     sum_len | min_len | avg_len | max_len |
| ---------------------- | ------ | ---- | -------- | ----------- | ------- | ------- | ------- |
| SRR26893485_1.fastq.gz | FASTQ  | DNA  |  919,704 | 219,258,464 |      35 |   238.4 |     251 |
| SRR26893485_2.fastq.gz | FASTQ  | DNA  |  919,704 | 219,377,153 |      35 |   238.5 |     251 |
| SRR26893959_1.fastq.gz | FASTQ  | DNA  |  791,743 | 188,558,779 |      35 |   238.2 |     251 |
| SRR26893959_2.fastq.gz | FASTQ  | DNA  |  791,743 | 188,654,783 |      35 |   238.3 |     251 |
| ecoli_1.fastq.gz       | FASTQ  | DNA  |   49,618 |  11,841,525 |      35 |   238.7 |     251 |
| ecoli_2.fastq.gz       | FASTQ  | DNA  |   49,618 |  11,847,005 |      35 |   238.8 |     251 |
| salmonella_1.fastq.gz  | FASTQ  | DNA  |  130,823 |  31,161,486 |      35 |   238.2 |     251 |
| salmonella_2.fastq.gz  | FASTQ  | DNA  |  130,823 |  31,177,052 |      35 |   238.3 |     251 |
| concat_1.fastq.gz      | FASTQ  | DNA  |  180,441 |  43,003,011 |      35 |   238.3 |     251 |
| concat_2.fastq.gz      | FASTQ  | DNA  |  180,441 |  43,024,057 |      35 |   238.4 |     251 |
| mixed_1.fastq.gz       | FASTQ  | DNA  |  180,441 |  43,003,011 |      35 |   238.3 |     251 |
| mixed_2.fastq.gz       | FASTQ  | DNA  |  180,441 |  43,024,057 |      35 |   238.4 |     251 |

## Skladanie do contigov, mapovanie na referenčnú databázu

Takto pripravené párové ready sa poskladali do contigov pomocou programu velvet:
Použili sme `.paired` súbory (hoci po shuffle sa obsahi `mixed_X.fastq.gz` a `mixed_X.paired.fastq.gz` zhodovali):

```sh
velveth velvet_output/ 31 -fastq.gz -separate mixed_1.paired.fastq.gz mixed_2.paired.fastq.gz 
velvetg velvet_output/ -ins_length 200 -min_contig_lgth 500 -scaffolding yes -read_trkg yes
```

Súbor `contigs.fa` (7.2 MB) sa použil pre mapovanie contigov na referenčnú databázy proteínov (`nr.dmnd`):

```sh
diamond blastx -d nr.dmnd -q velvet_output/contigs.fa -o mapped_contigs_long_read.m8 --long-reads --threads 8 --tmpdir dmnd-output/
```

## Metagenomická analýza v MEGAN

Súbor `mapped_contigs_long_read.m8` (320 MB) už potom možno použiť v softvéri MEGAN pre metagenomickú analýzu.

Po otvorení MEGAN sme na hlavnej lište použili `File > Import from BLAST`:
1. Pod kartou `Files`, v špecifikácii blast súboru sme vybrali náš výstup z DIAMOND: `mapped_contigs_long_read.m8` (format: BlastTab, Mode: `BlastX`).
2. Pod kartou `Files`, v špecifikácii MEGAN súboru sme len nastavili názov výstupného súboru: `mapped_contigs_long_read.rma6`
3. Pod kartou `Taxonomy`, sme zaškrtli možnosť `Analyze Taxonomy content`
    - ďalej sme klikli na `Load MeganMapDB mapping file` a vybrali súbor `megan-map-Feb2022.db`, ktorý možno stiahnuť z oficiálnej stránky Megan (9.3 GB)
    - potom sme ešte nad tým zaškrtli možnosť Fast Mode (čím sa odznačila možnosť Extended Mode)
4. Ostatné parametre (`LCA Params`) ostali ponechané, ostatné druhy analýz sme nechali vypnuté.

Táto analýza nám vytvorí súbor .rma6, ktorý potom možno načítať v MEGANe a prezerať si fylogenetický strom nášho metagenomického biómu.
Čísla odčítané z Meganu (ako reprezentatívnu hodnotu beriem popis uzlu s daným názvom, teda obsahuje aj všetky poduzly - poddruhy):
V archíve je .megan súbor obsahujúci hotový súpis, ktorý tiež možno prezerať. Ten sme vytvorili cez `File -> Export -> MEGAN Summary File`.

## Výsledky analýzy

| taxon                  | assigned | summed | expected ratio (norm.) | computed ratio (norm.) |
| ---------------------- | -------- | ------ | ---------------------- | ---------------------- |
| Escherichia coli       | 647      | 804    | 4  (1)                 | 804  (1)               |
| Salmonella             | 249      | 1833   | 12 (3)                 | 1833 (2.28)            |
| NCBI (all)             | 55       | 5693   | -                      | -                      |


Očakávaný pomer bol 4:12     (1:3) = 0.3333
Výsledný pomer bol: 804:1833 (1:2.28) = 0.4386

Je tam teda 10% odchýlka, čo nie je ideálne, ale to by sa dalo vysvetliť tým, že sme ako dĺžku genómu brali priemer z viacerých záznamov v NCBI databáze.
Pričom tento priemer mohol byť v prípade E.coli vyšší ako ten ideálny pre vzorku SRR26893485.
Alebo v prípade Salmonella sme mohli nájsť priemer takých vzoriek, ktorý vyšiel ako nižší než ten, ktorý by sa hodil pre vzorku SRR26893959.
Filtrovanie krátkych contigov VELVETom tiež mohlo prispieť k odchýlkam.

Pravdepodobne by sme dosiahli presnejšie výsledky ak by sme využili informácie z referenčného genómu organizmov E.coli a Salmonella.

