# Vizuálna analýza korelovaných mutácií v programe PyMOL

kurz: PBI - Pokročilá Bioinformatika
meno: Ján Maťufka  
mail: xmatuf00@stud.fit.vutbr.cz  
VUT ID: 222124  

## Obsah archívu:
- `msa.fasta` - výstup CLUSTALu s viacnásobným zarovnaním sekvencií (výsledok predošlého cvičenia), z ktorého vypočítali páry s korelovanými mutáciami
- `pairs.csv` - CSV súbor, ktorý sa použije na vizualizáciu (výstup skriptu `mutual_information.py`)
- `mutual_information.py` - Python skript, ktorý z daného viacnásobného zarovnania a ID sekvencie vyráta páry s korelovanými mutáciami na základe spoločnej informácie
- `visualize_correlated_mutations.py` - Python skript, ktorý na molekule danej PDB identifikátorom zobrazí páry korelovaných aminokyselín, ktoré sú uložené v CSV súbore
- `1AFR_A-CM-pairs.png` - exportovaný obrázok molekuly proteínu s vyznačenými korelovanými pármi

## Použitie:

Počas práce a testovania sa tieto príkazy spúšťali a fungovali z príkazového riadka operačného systému (nie pymolu).

Pre vytvorenie CSV s korelovanými pozíciami:
```shell
python3 mutual_information.py MSA_FASTA_FILE PDB_ID >CM_PAIRS_FILE_CSV
```

Pre vizualizáciu párov v PyMOLe:
```shell
python3 visualize_correlated_mutations.py PDB_ID CM_PAIRS_FILE_CSV
```

Konkrétne pre moje dáta:
```sh
python3 mutual_information.py msa.fasta 1afr_A >pairs.csv
python3 visualize_correlated_mutations.py 1afr_A pairs.csv
```

## Použité Python balíčky:
```
$ pip freeze
attrs==24.2.0
biopython==1.84
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
numpy==2.1.2
pymol==3.1.0a0
pyproject-toml==0.0.10
PyQt5==5.15.11
PyQt5-Qt5==5.15.15
PyQt5_sip==12.15.0
referencing==0.35.1
rpds-py==0.20.0
setuptools==75.2.0
toml==0.10.2
wheel==0.44.0
```
Python, verzia 3.12.3

## Popis Python skriptov:

Skript `mutual_information.py` vezme súbor vo FASTA formáte s viacnásobným zarovnaním podobných sekvencií (paralógy, ortológy), medzi ktorými je aj jedna sekvencia proteínu, pre ktorý existuje záznam v PDB.

Pre každú dvojicu pozícií tohto zarovnania spočíta hodnotu vzájomnej informácie (MI), čím vznikne matica -- nás zaujíma len polovica pod diagonálou, keďže na diagonále budú nuly a druhá polovica je symetrická.

Všetky dvojice pozícií sa zoradia podľa konkrétnej hodnoty MI a prvých 10 pozícií sa vypíše na štandardný výstup v CSV formáte.

Dôležité je podotknúť, že sa vypíšu len také hodnoty, pre ktoré má vo viacnásobnom zarovnaní proteínová sekvencia z PDB (daná argumentom `PDB_ID` pri spustení) jasnú hodnotu (a teda nie medzeru).

Zároveň sa tieto **pozície prepočítajú tak, aby korešpondovali nie s pozíciami v MSA, ale s pozíciami v sekvencii nášho konkrétneho proteínu**, na ktorom sa bude v ďalšom kroku vizualizovať.

Daný CSV súbor potom obsahuje pozície v MSA, ich korešpondujúce pozície v sekvencii vizualizovaného proteínu, a konkrétnu hodnotu MI.

----

Skript `visualize_correlated_mutations.py` načíta daný proteín na základe daného identifikátora, a z daného CSV súboru s pozíciami postupne nájde a vyznačí rôznymi farbami rôzne páry aminokyselín. Medzi nimi urobí úsečku v priestore, na ktorú pridá popis so vzdialenosťou (v Angstromoch) a s hodnotou MI pre daný pár.

Najprv sa ale postupným iterovaním nad alfa-uhlíkmi v sekvencii proteínu preložia pozície zo sekvencie proteínu na indexy reziduí v cif súbore, ktorý stiahne PyMOL. Toto je dôležité pre správnu vizualizáciu, keďže napríklad v prípade `1AFR_A` dotazy `select resi 1` až `select resi 18` neoznačia žiadne atómy. Náhľadom do CIF súboru sa zistilo, že konkrétne tento proteín má prvé reziduum na indexe 19. Kvôli robustnosti sa teda ešte vykoná tento medzikrok, čo zabezpečí **správne namapovanie indexov z predošlého kroku na aminokyseliny v štruktúre proteínu**.

Po vyznačení sa ešte vytvorí obrázok v aktuálnom adresári a pred skončením skriptu sa ešte pre vlastný pohľad nechá bežať PyMOL.
