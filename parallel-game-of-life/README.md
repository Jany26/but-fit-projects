# Hra Life v MPI

Pomocí knihovny MPI implementujte v C++ hru Life [1], [2]. 

Aktualizace zadání:
- 26.3.2024: doplněna sekce Diskuse, dotazy a sekce Odevzdání.
- 26.3.2024: doplněna informace o rozměrech mřížky
- 1.4.2024:
    - upřesněny minimální rozměry mřížky
    - upřesněno, že bude testována čtvercová mřížka
    - upraven formát výpisu
    - specifikováno okolí buňky
    - specifikováno chování za okrajem mřížky 
    - doplněna sekce Nápověda, poznámky
- 2.4.2024: přidána ukázka výstupu
- 3.4.2024: upravena sekce Odevzdání, přidán požadavek na odevzdání testovacího skriptu pro řešení pro usnadnění opravování
- 3.4.2024: do sekce Nápověda, poznámky přidána poznámka k vytvoření testovacího skriptu
- 6.4.2024: přidán příklad pro mřížku s pevnými stěnami
- 6.4.2024: přidán odkaz na online simulátor (do sekce Nápověda, poznámky)

## Hra Life

Hra Life reprezentuje příklad tzv. celulárního automatu. Hrací pole se skládá z buněk, které se v každém kroku přepínají mezi dvěma stavy: 

    živá (značíme 1)
    mrtvá (značíme 0)

Stavy buněk se v průběhu hry mění pomocí definované sady pravidel. Základní sada pravidel, kterou budete implementovat v projektu je následující:
- každá živá buňka s méně než dvěma živými sousedy umírá
- každá živá buňka se dvěma nebo třemi živými sousedy zůstává žít
- každá živá buňka s více než třemi živými sousedy umírá
- každá mrtvá buňka s právě třemi živými sousedy ožívá

Např. tato mřížka: 


```
00000000
00111000
01110000
00000000
```

bude mít po třech krocích hry s pravidly viz výše tvar:

```
01000000
00001000
01000000
00001000
```

pro implementaci typu wrap-around a 

```
00010000
01001000
01001000
00100000
```

pro implementaci s pevnými stěnami.

## Implementace

Program bude implementován v jazyce C++ s použitím MPI. Jako první argument bude program akceptovat název souboru s definicí hracího pole pro hru Life, jako druhý počet kroků, který má hra Life provést. 

Při implementaci zvolte okolí buňky jako tzv. osmi okolí (Moorovo okolí [3]), dle následujícího obrázku [3].

Moorovo okolí – Wikipedie

Buňky na za okrajem hracího pole ignorujte.

Program na standardní výstup (stdout) vypíše stav hracího pole pro hru Life po provedení zadaného počtu kroků. Ve výpisu budou označeny části mřížky vypočítané procesory jako: 
`ID: <část mřížky>`, kde:

    ID: rank procesoru
    <část mřížky>: část mřížky vyřešená procesorem s rankem ID

Tedy např.:

```
0: 00010000
0: 01001000
1: 01001000
1: 00100000
```

## Odevzdání

Projekt odevzdejte nejpozději 26.4.2024, 23:59:59 do StudISu. Odevzdává se pouze dobře okomentovaný zdrojový soubor life.cpp a testovací skript test.sh, který po spuštění přeloží váš program a spustí ho pro vámi definovaný počet procesů s parametry výše. Pokud testovací skript neodevzdáte, uveďte prosím počet procesů do hlavičky vašeho řešení. Testovací skript bude akceptovat dva argumenty:
- jméno souboru s definicí hracího pole pro hru Life.
- počet iterací, které má algoritmus provést

## Hodnocení projektu

V rámci projektu budu hodnotit zejména:
- zvolený přístup k paralelizaci hry Life
- dodržení zadání (tj. aby program dělal co má dle zadání, a vypisoval co má na stdout)
- kvalitu zdrojového kódu (komentáře, pojmenování proměnných a konstant, ... )

Při hodnocení budou ~~primárně~~ testovány čtvercové mřížky se sudým počtem řádků a sloupců s minimální velikostí mřížky 4x4.
Nápověda, poznámky

- dle [1] by měla být mřížka nekonečná, pro tento projekt ale můžete uvažovat mřížku s pevnou hranicí. Pokud budete uvažovat neomezenou mřížku, uveďte tuto skutečnost v hlavičce souboru s řešením, bude ohodnocena
- počet procesorů záleží na Vámi zvolenému přístupu k paralelizaci hry Life. Všechny rozumné počty budu uznávat. Za nerozumné považuji snad jedině řešení "každá buňka má vlastní procesor"
- cílem projektu není naprogramovat nejlepší paralelní implementaci hry Life, ani postihnout všechny možnosti, které mohou ve hře Life nastat. Cílem je, abyste si na praktickém problému vyzkoušeli, jak se dá podobná úloha paralelizovat pomocí MPI
- testovací skript můžete převzít ze zadání prvního projektu a upravit (odstranit generování čísel, upravit překlad a spuštění)
- hra Life je implementována na mnoha různých platformách, funkčnost vašeho řešení si lze vyzkoušet např na https://www.cuug.ab.ca/dewara/life/life.html

## Diskuse, dotazy

S dotazy se obracejte na iveigend@fit.vut.cz , případně lze využít i diskusní fórum pro dotazy.

[1] https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life  
[2] https://web.stanford.edu/class/sts145/Library/life.pdf  
[3] https://en.wikipedia.org/wiki/Moore_neighborhood  
