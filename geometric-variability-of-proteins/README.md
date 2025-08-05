# Projekt -- Analýza geometrickej variability proteínových štruktúr

kurz: PBI - Pokročilá Bioinformatika  
meno: Ján Maťufka  
mail: xmatuf00@stud.fit.vutbr.cz  
VUT ID: 222124  

## Zadanie projektu -- Štruktúra proteínu

Vizualizujte na štruktúre proteínu X geometrickú variabilitu aminokyselín (na odpovedajúcich si pozíciách) v rámci rodiny proteínov, do ktorej proteín X patrí v hierarchii CATH.
Môžete vizualizovať podľa svojho výberu napríklad odchýlku C-alfa uhlíkov po zarovnaní štruktúr alebo uhly bočného reťazca voči najbližším susedom, prípadne inú obdobnú geometrickú vlastnosť.

## Obsah archívu

- `prepare_data.py` - obsahuje funkcie, ktoré sa starajú o vyhľadanie CATH rodiny zadaného proteínu, stiahnutie potrebných dát a extrakciu chainov z PDB súborov
- `compute_ca_angles.py` - obsahuje funkcie, ktoré vykonajú zarovnanie proteínových štruktúr, extrakciu geometrických dát (pozície C-alfa uhlíkov) a výpočet uhlov
- `visualize_deviations.py` - štartovací bod programu, spustí PyMOL a použije dáta o variabilite uhlov pre grafické zobrazenie
- `requirements.py` - zoznam Python balíčkov, ktoré je nutné nainštalovať pre fungovanie projektu
- `1mbn-output.png` - obrázok obsahujúci grafický výstup v PyMOLe pre testovaný proteín 1MBN (rodina globínov)

## Princíp fungovania projektu

Projekt berie len jeden argument -- PDB identifikátor proteínu (4 alfanumerické znaky), pre ktorý sa bude počítať geometrická variabilita medzi uhlami alfa uhlíkov.
Teda príklad spustenia: `python3 visualize_deviations.py 1mbn`

Z nasledujúcej stránky [CathDB](http://www.cathdb.info/wiki?id=data:index) sa prostredníctvom FTP z odkazu [OregonFTP BioChem](ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/daily-release/newest/cath-b-newest-all.gz) stiahne CATH databáza, ktorá informácie o všetkých proteínoch a ich príslušných rodinách (teda ich 4 číselné hierarchické identifikátory oddelelené bodkami).
Použitím `gunzip` sa extrahuje do súboru `cath-b-newest-all`. 

Pomocou nástroja `grep` sa z databázy vyberú relevantné riadky obsahujúce názvy proteínových reťazcov ("chains" -- 1 proteín sa môže skladať z viacerých reťazcov -- v tomto prípade sa už jedná o kvartérnu štruktúru, reťazce v rámci 1 proteínu môžu patrieť do viacerých rodín), ktoré patria do rovnakej rodiny ako zadaný proteín.

V prípade, že zadaný proteín obsahuje viacero reťazcov, celá procedúra sa spustí pre každý jeho reťazec zvlášť, čím sa vypočítajú odchýlky pre celú proteínovú štruktúru.

Všetky proteíny z danej rodiny sa najprv stiahnu pomocou `wget`, potom sa z nich extrahujú jednotlivé "reťazce" (pomocou `PDBParser` a `PDBIO` z balíčka `Bio.PDB`).
Z PDB súborov extrahovaných reťazcov sa vyberú súradnice alfa uhlíkov (záznamy `ATOM`, kde 3. stĺpec tvorí reťazec `CA`). 
Takto dostaneme pole súradníc alfa uhlíkov (kostra proteínu), a nad týmto poľom môžeme vypočítať všetky uhly medzi nimi (pre prvú a poslednú pozíciu nevieme zistiť uhol).

Po extrakcii atómov a vypočítaní uhlov sa vykoná pomocou nástroja `TMalign` párové zarovnanie každého reťazca v rodine so zadaným proteínom s príkazovej riadky.
Z výstupu programu `TMalign` sa na základe zarovnaných aminokyselín extrahujú relevantné uhly a uložia do matice (1 riadok matice reprezentuje uhly medzi alfa uhlíkmi v rámci 1 proteínu, stĺpec matice reprezentuje uhly alfa uhlíkov na konkrétnej pozícii zarovnanej k referenčnému proteínu).

Z matice uhlov sa vypočíta pre každý stĺpec odchýlka, tá sa potom v rámci všetkých odchýlok normalizuje (Z-score) a tieto normalizované (absolútne) hodnoty (rozsah 0 až 5 približne) sa rozdelia do intervalov, každý interval má priradenú vlastnú farbu.
Teda atómy alfa uhlíkov, ktoré so susedmi zvierajú uhly v odchýlke v danom (normalizovanom) rozsahu, budú mať priradenú farbu nasledovne:
- rozsah `< 0   ; 0.5 )` - zelená farba
- rozsah `< 0.5 ; 1   )` - žltá farba
- rozsah `< 1   ; 1.5 )` - oranžová farba
- rozsah `< 1.5 ; 2   )` - lososová farba
- rozsah `< 2   ; inf )` - červená farba

Rozdelenie farieb by malo odrážať 3-sigma pravidlo, teda približne 68 % C-alfa atómov bude zelených alebo žltých, približne 27 % C-alfa atómov bude oranžových alebo lososových, a posledných 5 % bude červených.
Červené alfa uhlíky reprezentujú také miesta v proteínovej štruktúre, u ktorých vo veľkej miere dochádza ku geometrickej variabilite v rámci CATH rodiny.

Program počas výpočtu nainicializuje `PyMOL`, po výpočte sa v ňom objaví zadaný proteín s farebne vyznačenými alfa uhlíkmi podľa ich geometrickej variability.

## Testovacie prostredie a požiadavky

Sťahovanie proteínov je najdlhšia časť celého výpočtu. Ak nebudeme do výpočetného času počítať sťahovanie (a extrakciu reťazcov), potom:
- pre proteín 1MBN -- globín (obsahuje len 1 chain -- celá rodina globínov obsahuje 1273 proteínov, 3059 "chainov") trvá zarovnanie, výpočet a vizualizácia približne 90-120 sekúnd.
- pre proteín 5X2R (12 chainov -- všetky patria do globínov, teda 3059x12 zarovnaní) -- výpočet trvá približne 18 minút.

Testované na:
OS: Ubuntu 24.04.1 LTS x86_64
CPU: Intel i7-8750H (12) @ 4.100GHz
RAM: 16 GB
Python 3.12.3

Zoznam Python balíčkov (bez závislostí), je v súbore `requirements.txt`.
V projekte sa ďalej používajú UNIX nástroje `grep`, `wget`,  `gunzip`.


## Nastavenie TMalign pre správne fungovanie

Pre zarovnávanie proteínových sekvencií (na základe čoho sa určí, ako sa budú počítať odchýlky uhlov), je potrebné mať nainštalovaný program `TMalign` (testované na verzii 20190425).
Pre inštaláciu `TMalign` nasledujte inštrukcie na [tejto stránke](https://zhanggroup.org/RNA-align/TMalign/).
Cesta pre binárku/spustiteľný súbor je v programe určená konštantou, pre správne fungovanie ju možno bude potrebné upraviť (`visualize_deviations.py`, premenná `TMALIGN_PATH`).
Napríklad v prípade, že binárka ostane v adresári s Python skriptami, nastavte túto konštantu na `./TMalign`.
Program určite funguje na Linuxe, inštrukcie pre Windows na stránke nie sú, preto fungovanie projektu na Windowse nie je zaručené.
