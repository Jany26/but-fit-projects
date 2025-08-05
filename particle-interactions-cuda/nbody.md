# PCG projekt 1

- autor: xmatuf00  (Ján Maťufka)

## Měření výkonu (čas / 100 kroků simulace)

### Průběžné
|   N   | CPU [s]  | Step 0 [s] | Step 1 [s] | Step 2 [s] | Step 3 [s] | Step 4 [s] |
|:-----:|----------|------------|------------|------------|------------|------------|
|  4096 | 0.492139 | 0.356849   | 0.324175   | 0.281880   | 0.281439   | 0.280151   |
|  8192 | 1.471328 | 0.716745   | 0.648957   | 0.561706   | 0.560862   | 0.556953   |
| 12288 | 2.478942 | 1.076999   | 0.974800   | 0.840639   | 0.839526   | 0.833745   |
| 16384 | 3.386801 | 1.434922   | 1.299459   | 1.120183   | 1.118646   | 1.110591   |
| 20480 | 5.059240 | 1.794499   | 1.624812   | 1.400074   | 1.397916   | 1.387358   |
| 24576 | 7.112179 | 2.152324   | 1.950841   | 1.680744   | 1.678265   | 1.664112   |
| 28672 | 9.892856 | 2.506717   | 2.259562   | 1.956935   | 1.954375   | 1.940908   |
| 32768 | 12.59829 | 2.848809   | 2.575775   | 2.231653   | 2.218883   | 2.217745   |
| 36864 | 15.54297 | 3.203261   | 2.897456   | 2.512116   | 2.496370   | 2.494536   |
| 40960 | 19.36099 | 3.553905   | 3.218326   | 2.782240   | 2.774911   | 2.771297   |
| 45056 | 23.48723 | 3.908683   | 3.539725   | 3.049871   | 3.049115   | 3.048144   |
| 49152 | 27.69359 | 4.263781   | 3.868938   | 3.326676   | 3.325628   | 3.324946   |
| 53248 | 32.63063 | 4.618734   | 4.182997   | 3.602853   | 3.602598   | 3.601736   |
| 57344 | 37.43660 | 8.068312   | 7.106498   | 6.546916   | 6.540938   | 6.540245   |
| 61440 | 42.85863 | 8.644032   | 7.613436   | 7.013109   | 7.007908   | 7.007129   |
| 65536 | 49.46104 | 9.230069   | 8.121534   | 7.474740   | 7.474319   | 7.474397   |
| 69632 | 55.14939 | 9.822344   | 8.629131   | 7.942009   | 7.942496   | 7.942741   |
| 73728 | 62.04446 | 10.385945  | 9.136297   | 8.409489   | 8.408735   | 8.407881   |
| 77824 | 69.26138 | 10.961464  | 9.643895   | 8.875566   | 8.875375   | 8.875056   |
| 81920 | 76.60071 | 11.538116  | 10.150876  | 9.342993   | 9.341788   | 9.341704   |

Pozn: Meranie bolo uskutočnené na výpočetnom uzle acn01.
Iné uzly môžu mať rýchlejšie časy, alebo aj pomalšie.

### Závěrečné
|    N   |  CPU [s] | GPU [s] | Zrychlení | Propustnost [GiB/s] | Výkon [GFLOPS] |
|:------:|:--------:|:-------:|:---------:|:-------------------:|:--------------:|
|   1024 |   1.0928 |  0.0764 |   14.30 x |         0.052116811 |         96.404 |
|   2048 |   0.5958 |  0.1467 |    4.06 x |         0.045439228 |        196.177 |
|   4096 |   0.6652 |  0.2875 |    2.31 x |         0.042058527 |        396.229 |
|   8192 |   1.6599 |  0.5681 |    2.92 x |         0.040326267 |        796.232 |
|  16384 |   3.3655 |  1.1305 |    2.98 x |         0.039450824 |      1 596.034 |
|  32768 |  12.7233 |  2.2580 |    5.63 x |         0.039022416 |      3 196.048 |
|  65536 |  48.9732 |  7.5493 |    6.49 x |         0.023031607 |      3 791.992 |
| 131072 | 195.9965 | 22.4658 |    8.72 x |         0.015404075 |      5 084.590 |

Pozn: ncu profiling report vypísal Memory Throughput v jednotkách MByte/s.
GiB/s sa dostal z MByte/s takto: MByte/s * 1000^2 / 1024^3 = GiB/s

## Otázky

### Krok 0: Základní implementace
**Vyskytla se nějaká anomále v naměřených časech? Pokud ano, vysvětlete:**

V nameraných časoch sa vyskytla anomália.
Medzi N = 53248 a N = 57344 možno pozorovať takmer 2x spomalenie.
Predtým aj potom čas rastie lineárne.

Na Karolíne sú NVIDIA A100 GPU.
V prezentáciách PCG (týždeň 4, slajd 35) je napísané,
že A100 NVIDIA GPU majú 108 SM procesorov.
Dané testy sa spúšťali s nastavením 512 vlákien na blok.
512x108 = 55296, čo je presne medzi 53248 a 57344.
Teda pre vstup N = 57344 už je viac blokov ako SM jednotiek,
a preto narástla režia spojená s plánovaním a synchronizáciou,
čo by vysvetľovalo značné spomalenie práve medzi týmito veľkosťami.

### Krok 1: Sloučení kernelů
**Došlo ke zrychlení?**
Áno, približne o 10-15 %.

**Popište hlavní důvody:**
V každom z pôvodných 3 kernelov sa niekde na začiatku pristupovalo
do globálnej pamäte a na konci niekam do globálnej pamäte ukládalo.
Zlúčením kernelov už dané hodnoty má vlákno uložené niekde v registri a
pristupuje rovno tam, čím sa značne zníži réžia spojená s globálnou pamäťou.

Zároveň sa výpočty gravitačnej a kolíznej rýchlosti zlúčia do jedného cyklu,
čo tiež prispieva k malému zlepšeniu.

Réžia so spúšťaním a ukončovaním kernelov sa tiež značne zníži
pri spúšťaní jedného kernelu miesto troch.


### Krok 2: Sdílená paměť
**Došlo ke zrychlení?**

Áno, približne o 10-15 %, podobne ako to bolo so zlúčením kernelov.

**Popište hlavní důvody:**

Keďže každé vlákno prepočítavá pre svoju časticu vplyv gravitačných a kolíznych
síl so všetkými ostatnými časticami, musí N-krát pristúpiť do globálnej pamäte.
Avšak každý blok môže na začiatku všetky svoje častice uložiť do zdieľanej pamäte,
a potom si každá častica prepočíta vplyvy častíc z daného bloku.
Prístup do globálnej pamäte v tomto cykle sa tak zníži z N na počet blokov (gridDim).
V ostatných prípadoch sa pracuje so zdieľanou pamäťou,
ktorá je značne rýchlejšia ako globálna.

### Krok 5: Měření výkonu
**Jakých jste dosáhli výsledků?**

Pre všetky vstupy je GPU implementácia niekoľkonásobne rýchlejšia oproti CPU.
Na základe trendu v dátach sa dá očakávať, že zrýchlenie bude ešte viac narastať
pre väčšie vstupy.

**Lze v datech pozorovat nějaké anomálie?**

Áno, spočiatku sa dá pozorovať pokles zrýchlenia (N = 1024 -> N = 2048 -> N = 4096).
Až potom od veľkosti 4096 vyššie začne znovu zrýchlenie narastať.
To bude spôsobené tým, že CPU implementácia má namerané spočiatku klesajúce časy.

Zaujímavejšie je ale sledovať ako rýchlo rastú výpočetné časy GPU.
Do veľkosti 32K rastú lineárne (2x väčší vstup -> 2x dlhší runtime).
Od veľkosti 32K to ale rastie vo väčšej miere (2x väčší vstup -> cca 3x dlhší runtime).
To si myslím bude z podobných dôvodov ako je popísané v kroku 1.
Akurát medzi 65K a 131K sa prejde cez ďalší prah (a to konkrétne N = 110592,
čo znamená 216 blokov, teda 2 bloky na 1 SM jednotku), a teda budú 3 bloky na niektoré SM jednotky.

Ešte sa môže zdať zvláštne, že priepustnosť pamäte klesá, ale to je kvôli tomu, že množstvo
výpočtov rastie kvadraticky (vplyv každej častice s každou), a počet častíc len lineárne.
Teda väčšia časť behu programu počíta a menšia pracuje s pamäťou.
