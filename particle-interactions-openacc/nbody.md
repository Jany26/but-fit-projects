# PCG projekt 2
- autor: xmatuf00

## Měření výkonu (čas / 100 kroků simulace)

### Průběžné
|   N   | CPU [s]  | Step 0 [s] | Step 1 [s] | Step 2 [s] | Step 3 [s] |
|:-----:|----------|------------|------------|------------|------------|
|  4096 | 0.492139 |  0.139580  |  0.079456  |  0.079574  |  0.106941  |
|  8192 | 1.471328 |  0.276228  |  0.156029  |  0.156116  |  0.183977  |
| 12288 | 2.478942 |  0.412488  |  0.232611  |  0.232709  |  0.261158  |
| 16384 | 3.386801 |  0.579476  |  0.335939  |  0.336024  |  0.364116  |
| 20480 | 5.059240 |  0.723535  |  0.419409  |  0.419560  |  0.448922  |
| 24576 | 7.112179 |  0.867727  |  0.502824  |  0.502990  |  0.533755  |
| 28672 | 9.892856 |  1.025172  |  0.586709  |  0.586973  |  0.619766  |
| 32768 | 12.59829 |  1.187226  |  0.685625  |  0.685776  |  0.716815  |
| 36864 | 15.54297 |  1.335032  |  0.771879  |  0.772029  |  0.802758  |
| 40960 | 19.36099 |  1.483058  |  0.856610  |  0.856571  |  0.888556  |
| 45056 | 23.48723 |  1.929969  |  1.191810  |  1.192092  |  1.225222  |
| 49152 | 27.69359 |  2.106072  |  1.301087  |  1.301841  |  1.337620  |
| 53248 | 32.63063 |  2.281681  |  1.410913  |  1.411391  |  1.447410  |
| 57344 | 37.43660 |  3.106473  |  1.890845  |  1.890913  |  1.930648  |
| 61440 | 42.85863 |  3.332999  |  2.025342  |  2.026572  |  2.067269  |
| 65536 | 49.46104 |  3.555841  |  2.159996  |  2.160546  |  2.202872  |
| 69632 | 55.14939 |  4.304934  |  2.686013  |  2.686064  |  2.734004  |
| 73728 | 62.04446 |  4.674383  |  2.868587  |  2.868770  |  2.914477  |
| 77824 | 69.26138 |  4.932216  |  3.026516  |  3.026853  |  3.074748  |
| 81920 | 76.60071 |  5.191756  |  3.186349  |  3.186660  |  3.235320  |

(namerané na uzle acn09)

### Závěrečné
|    N   |  CPU [s] | GPU [s]  | Zrychlení | Propustnost [GiB/s] | Výkon [GFLOPS] |
|:------:|:--------:|:--------:|:---------:|:-------------------:|:--------------:|
|   1024 |   1.0928 | 0.056784 | 19.24x    |  0.122273341        |   118.584      |
|   2048 |   0.5958 | 0.077221 |  7.72x    |  0.122552738        |   237.583      |
|   4096 |   0.6652 | 0.118204 |  5.63x    |  0.119805336        |   469.538      |
|   8192 |   1.6599 | 0.199748 |  8.31x    |  0.119078904        |   933.759      |
|  16384 |   3.3655 | 0.387522 |  8.68x    |  0.109244138        | 1 714.096      |
|  32768 |  12.7233 | 0.753533 | 16.88x    |  0.106561929        | 3 345.738      |
|  65536 |  48.9732 | 2.273101 | 21.54x    |  0.067232177        | 4 233.605      |
| 131072 | 195.9965 | 9.245743 | 21.20x    |  0.035017729        | 4 332.148      |

(namerané na uzle acn08)

## Otázky

### Krok 0: Základní implementace
**Vyskytla se nějaká anomále v naměřených časech? Pokud ano, vysvětlete:**

Rýchlosť škáluje lineárne, okrem troch skokov pri veľkostiach N = 45056, N = 57344 a N = 69632.
Tieto skoky nastávajú približne každé 3-4 kroky.

Vysvetlení môže byť viacero.
Podobne ako pri 1. projekte sa môže jednať o saturáciu SM jednotiek NVIDIA A100 procesorov.
V predošlom projekte sa pri veľkosti 55296 saturovalo 108 SM procesorov (512 vlákien na 1).
Keďže pri OpenACC nevieme, ako prebieha plánovanie a rozdeľovanie zdrojov, vysvetlení môže byť viacero:
- neoptimálna voľba počtu vlákien na blok,
- rozdelenie lokálnej pamäte/registrov medzi vlákna
- nárast réžie s plánovaním (viac logických blokov na SM procesor)


### Krok 1: Sloučení kernelů
**Došlo ke zrychlení?**

Áno, približne o 40%. (konzistentne pre všetky vstupy, napr. pri najväčšom vstupe sme z 5.2s spadli na 3.2s).

**Popište hlavní důvody:**

Menej presúvania dát, zlučovanie niektorých výpočtov. Miesto 3 for cyklov sa koná len 1.

### Krok 2: Výpočet těžiště
**Kolik kernelů je nutné použít k výpočtu?**

Stačí jeden (centerOfMass).

**Kolik další paměti jste museli naalokovat?**

Iba 4 floaty (16B) pre výsledok (teda jediný float4 element comBuffer).

**Jaké je zrychelní vůči sekveční verzi? Zdůvodněte.** *(Provedu to smyčkou #pragma acc parallel loop seq)*

|   N   | reduction [s] | sequential [s] |
|:-----:|---------------|----------------|
|  4096 |   0.000112    |    0.000995    |
|  8192 |   0.000108    |    0.003429    |
| 12288 |   0.000114    |    0.002764    |
| 16384 |   0.000115    |    0.003658    |
| 20480 |   0.000111    |    0.004567    |
| 24576 |   0.000122    |    0.005472    |
| 28672 |   0.000122    |    0.006408    |
| 32768 |   0.000124    |    0.007283    |
| 36864 |   0.000120    |    0.008182    |
| 40960 |   0.000131    |    0.009088    |
| 45056 |   0.000119    |    0.009972    |
| 49152 |   0.000123    |    0.010859    |
| 53248 |   0.000131    |    0.011766    |
| 57344 |   0.000133    |    0.012666    |
| 61440 |   0.000127    |    0.013542    |
| 65536 |   0.000128    |    0.014460    |
| 69632 |   0.000145    |    0.015364    |
| 73728 |   0.000140    |    0.016257    |
| 77824 |   0.000142    |    0.017174    |
| 81920 |   0.000144    |    0.018067    |

(uzol acn08)

Pri najmenšom vstupe sa jedná o 8.88x zrýchlenie.
Pri najväčšom vstupe sa jedná o 125.46x zrýchlenie

Sekvenčná verzia využíva len jedno vlákno na výpočet ťažiska. Pred spracovaním
ďalšieho prvku sa musí počkať na výsledok.

Paralelná verzia s redukciou využíva viacero vlákien a hierarchický spôsob redukcie.
Teda škáluje logaritmicky, preto výpočetný čas takmer vôbec nerastie.

### Krok 4: Měření výkonu
**Jakých jste dosáhli výsledků?**

Pre všetky vstupy je GPU implementácia niekoľkonásobne rýchlejšia oproti CPU.
Na základe trendu v dátach sa dá očakávať, že zrýchlenie bude ešte viac narastať
pre väčšie vstupy.

**Lze v datech pozorovat nějaké anomálie?**

Áno, spočiatku sa dá pozorovať pokles zrýchlenia (N = 1024 -> N = 2048 -> N = 4096).
Až potom od veľkosti 4096 vyššie začne znovu zrýchlenie narastať.
To bude spôsobené tým, že CPU implementácia má namerané spočiatku klesajúce časy.

Priepustnosť pamäte klesá, čo ale dáva zmysel, keďže sa na vstup rastie lineárne a výpočet kvadraticky.
Menej času sa teda pracuje s pamäťou.
