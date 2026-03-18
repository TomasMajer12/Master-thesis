

Vystup:



Sudoku data:


    Kaggle dataset + train 60 20 20 procent
    udelat obrazky - symbolicka a vizualni
    mnist - rucne psana cisla se symboli
    z mnistu brat random samples - stane se hodne cisel kterych je stejny, dobry na trenovaci nic moc na testovaci , train/test/val, 
    obrazky nahodne vybirat - udelat potom histogram

    Zkusit najit jinou i jinou sadu
    --> Vytvorit data symbolicka i vizualni

    vytvorit statisticky - kolik prikladu je v kazde sade, histogram



Algoritmus a inference pro pripady
struktura je retezec + inference pomoci dynamickeho prog.

struktura je retezec + inference je pomoci lin prog. --> overit 

obecna struktura + inference pomoci lin prog.

Overeni ohledne funcnosti

Benchmark - symbolicke a vizualni

finalni metrika -> vyhodnoceni na validacnich datech --> na trvdo zafixovat, nedelat crossvalidaci



benchmark - testovaci data 10k, validacni 10k, trenovaci 30k
slozitost sudoku - zatim nahodny vyber


hmc - vzit stejne pravdepodobnosti a stejnou delku sekvence

benchmark pro HMC, test 20k, val 20k, train 60k - varianta symbolicke a obrazky

Evaluace na benchmarku:
pro kazdou variantu algoritmu

metriky - testovaci chyba, 0/1 ztrata, hamming_loss - normalizovana na pocet labelu, 
udelat ucici krivky - pocet trenovacich dat a testovaci chyba a trenavaci chyba


obrys:

jak bude vypadat inteface

yaml - struktura NN a klasifikatoru
hyperparamatry

udelat jeden konfuguracni soubor na cesty na data, hyperparametry, architektura

udelat definice tohoto souboru

jeden - trenovani, uceni, architektura
druhy - samotnou architekturu

varianta jedna - kazdy symbol ma svuj fixni obrazek - vstup prediktoru je sekvence obrazku, 
druha varianta - vstup je jeden obrazek

pouzij (resnet50) architekturu, vlastni

jeden z parametru bude backbone z pytorch - neco co uz tam je naimplementovane

jaka je struktura retezce?
pocet symbolu na vystupu a stavu - delka 30 a kazdy symbol muze mit 10 vystupu

vstup - symbolicky / obrazkovy

plna varianta - jaky je backbone - graf sousednosti, kolik ma label symbolu a v seznamu hrany

je vstup sekvence obrazku a nebo jeden obrazek

A - input is sequece
B - imput is symbolic
C - 

(vstup bude 20 obrazku s cisli nebo obrazek bude jeden a bude muset udelat segmentaci)


baseline:
natrenovat OCR, ktere pri inferenci pustim nad kazdou cislici a normalnim sudoku solverem
staci jenom pro sudoku


TODO:
DATA 
Udelat inferenci s daty i s obrazku - sekvence obrazku a jeden obrazek

