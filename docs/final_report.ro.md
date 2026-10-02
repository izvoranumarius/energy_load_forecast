[English](final_report.md)

# Prognoza consumului de energie electrică: raport final

## 1. Obiectivul proiectului

Proiectul realizează prognoza cererii de energie electrică pentru România folosind date istorice orare de consum de pe ENTSO-E Transparency Platform.

Problema de prognoză este definită astfel:

- **Geografie:** România (`RO`)
- **Țintă:** Actual Total Load (`load_mw`)
- **Granularitate:** orară
- **Orizont de prognoză:** 24 de ore înainte
- **Tipul prognozei:** prognoză punctuală, cu un interval opțional P10-P90

Comparația principală între modele include două baseline-uri simple, LightGBM și N-BEATS.

## 2. Date și ingestie

Datele istorice privind consumul de energie electrică sunt colectate din două surse suportate de proiect:

1. exporturi CSV din Transparency Platform;
2. API-ul ENTSO-E prin `entsoe-py`.

Stratul de ingestie normalizează timestamp-urile la UTC și resamplează datele sursă la o serie orară `load_mw`. Valorile lipsă sunt păstrate în mod intenționat pentru etapa de preprocessing, în loc să fie completate automat.

Pentru experimentul final, setul de date acoperă perioada **iulie 2025 până în decembrie 2025**.

## 3. Calitatea datelor și preprocessing

Pipeline-ul de preprocessing verifică principalele probleme structurale relevante pentru prognoza orară a consumului:

- tratarea fusului orar și normalizarea la UTC;
- timestamp-uri duplicate;
- ordinea cronologică;
- valori lipsă și NaN-uri de la finalul seriei, asociate întârzierii publicării;
- valori negative ale consumului;
- frecvența orară regulată;
- diferența dintre valorile lipsă din interiorul seriei și cele de la final.

NaN-urile de la final sunt eliminate deoarece corespund cozii incomplete a datelor publicate. NaN-urile din interior sunt tratate drept probleme de calitate a datelor și nu sunt ignorate în mod automat.

Datele sunt împărțite cronologic în ferestre de train, validation și test. Experimentul final rezervă o fereastră completă de 24 de ore pentru validation și o fereastră completă de 24 de ore pentru test.

## 4. Analiză exploratorie și sezonalitate

EDA analizează:

- seria de consum în timp;
- consumul mediu în funcție de ora locală;
- consumul mediu în funcție de ziua săptămânii;
- diferența dintre zilele lucrătoare și weekend;
- distribuția consumului și boxplot-ul;
- consumul mediu lunar;
- profilul de consum zi a săptămânii × oră;
- autocorelația la lagurile de 24 de ore și 168 de ore.

Un profil orar repetitiv susține existența unei componente de sezonalitate zilnică, iar diferențele dintre zilele lucrătoare și weekend susțin existența unei componente de sezonalitate săptămânală.

Autocorelația măsurată este **0.803 la 24 de ore** și **0.855 la 168 de ore**. Aceste valori oferă o evidență numerică a persistenței zilnice și săptămânale ridicate în serie.

### Limitare importantă privind sezonalitatea anuală

Setul de date final disponibil acoperă doar perioada iulie-decembrie 2025. Prin urmare, profilul lunar este util ca o comparație descriptivă în cadrul perioadei disponibile, dar nu trebuie prezentat ca o estimare fiabilă a sezonalității anuale complete. Pentru o concluzie solidă privind sezonalitatea anuală ar fi necesare mai multe cicluri anuale complete.

Notebook-ul final de EDA conține graficele de sezonalitate și verificările de autocorelație.

## 5. Feature engineering

Modelul LightGBM folosește doar caracteristici derivate din informații disponibile înainte de momentul prognozei:

- `hour_local`
- `day_of_week_local`
- `is_weekend_local`
- `month_local`
- `load_mw_lag1`
- `load_mw_lag24`
- `load_mw_lag168`
- `load_mw_rolling_mean_3h`
- `load_mw_rolling_mean_24h`

Caracteristicile rolling folosesc un shift de un pas înainte de calcularea mediei, astfel încât valoarea curentă a țintei să nu fie inclusă.

În timpul prognozei multi-step pentru validation și test, caracteristicile LightGBM sunt construite recursiv. Valorile reale indisponibile din viitor sunt înlocuite cu valorile prognozate, ceea ce previne introducerea valorilor reale viitoare ale consumului în forecast.

## 6. Modelele de prognoză

### Baseline 1 - Naive

Prognoza Naive repetă ultima valoare observată a consumului pentru toate cele 24 de ore din orizont.

### Baseline 2 - Seasonal Naive

Prognoza Seasonal Naive repetă cel mai recent profil de consum de 24 de ore. Aceasta oferă o referință simplă bazată pe sezonalitatea zilnică.

### LightGBM

LightGBM este folosit ca principal model de machine learning bazat pe caracteristici. Un model punctual este antrenat pentru prognoze punctuale, iar modele separate pe cuantile sunt antrenate pentru cuantilele 10%, 50% și 90%, pentru intervalul opțional de predicție.

### N-BEATS

N-BEATS este folosit ca o a doua familie de modele, care învață direct din seria istorică a consumului, fără tabelul de caracteristici construit pentru LightGBM. Seria de antrenament este scalată folosind un scaler ajustat numai pe datele de train.

## 7. Metodologia de evaluare

Prognozele punctuale sunt evaluate folosind:

- **MAE** - eroarea absolută medie;
- **RMSE** - rădăcina pătrată a erorii pătratice medii;
- **MAPE** - eroarea procentuală absolută medie, interpretată cu atenție atunci când valorile reale sunt foarte mici.

Intervalele de predicție sunt evaluate folosind:

- pinball loss pentru P10;
- pinball loss pentru P90;
- acoperire empirică;
- lățimea medie a intervalului.

Comparația finală pe test este realizată pe același index de test de 24 de ore pentru toate modelele.

## 8. Analiza erorilor și interpretabilitate

Proiectul include:

- consum real versus consum prognozat;
- reziduuri în timp;
- distribuția reziduurilor;
- eroarea absolută medie în funcție de ora locală;
- cele mai mari erori absolute de prognoză;
- importanța caracteristicilor pentru LightGBM.

Graficul de importanță a caracteristicilor pentru LightGBM arată că `hour_local` și `load_mw_lag1` au cea mai mare contribuție la model. `load_mw_lag168` și media rolling pe 3 ore sunt, de asemenea, importante, urmate de media rolling pe 24 de ore și `load_mw_lag24`. Caracteristicile calendaristice precum `month_local` și `day_of_week_local` contribuie mai puțin, în timp ce `is_weekend_local` nu primește importanță măsurată în acest model antrenat.

Graficul reziduurilor arată erori negative mai mari în primele ore ale zilei, urmate de valori mai apropiate de zero și de reziduuri pozitive în zona după-amiezii. Spre seară, reziduurile devin din nou mixte. Deoarece analiza folosește o singură fereastră de test de 24 de ore, aceste observații trebuie interpretate ca rezultate pentru acest orizont particular și nu ca un comportament general pentru întregul set de date.

## 9. Rezultate finale

Comparația cantitativă finală se bazează pe fereastra de test de 24 de ore din **2025-12-31 00:00:00 UTC** până la finalul zilei.

### Tabelul rezultatelor

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Naive | 773.24 | 975.43 | 11.2704 |
| Seasonal Naive | 199.375 | 256.484 | 3.02983 |
| LightGBM | 230.052 | 266.821 | 3.64566 |
| N-BEATS | 300.971 | 352.976 | 4.73213 |

Pe această fereastră de test, baseline-ul Seasonal Naive are cele mai mici valori pentru MAE, RMSE și MAPE dintre modelele evaluate. LightGBM urmează cu erori relativ apropiate, în timp ce N-BEATS are erori mai mari pe acest orizont particular de 24 de ore. Rezultatul arată că o referință simplă bazată pe sezonalitatea zilnică poate rămâne foarte informativă pentru acest set de date și această perioadă de test.

### Interval de predicție P10-P90

Modelele LightGBM pe cuantile au produs următoarele rezultate:

| Metrică | Valoare |
|---|---:|
| Pinball loss P10 | 39.7004 |
| Pinball loss P90 | 59.5282 |
| Acoperire empirică | 87.5% |
| Lățimea medie a intervalului | 922.193 MW |

Acoperirea empirică este calculată pe aceeași fereastră de test de 24 de ore, astfel încât trebuie interpretată cu atenție deoarece eșantionul este mic.

### Comparație vizuală

Graficul final de comparație arată consumul real împreună cu cele patru prognoze punctuale pe orizontul de test de 24 de ore. Modelul Naive rămâne aproape constant, în timp ce Seasonal Naive urmărește mai bine forma zilnică. LightGBM și N-BEATS reproduc ambele tiparul principal intrazilnic, cu abateri diferite în zona dimineții și a după-amiezii.

## 10. Structura proiectului

Notebook-urile documentează fluxul de analiză și experimentare, iar modulele Python conțin implementarea reutilizabilă extrasă din această muncă.

Notebook-uri principale:

- `00_explore_raw_csv.ipynb`
- `01_api_ingestion.ipynb`
- `02_preprocessing_exploration.ipynb`
- `03_eda.ipynb`
- `04_baselines.ipynb`
- `05_feature_engineering.ipynb`
- `06_lightgbm.ipynb`
- `07_nbeats.ipynb`
- `08_error_analysis.ipynb`
- `09_final_results.ipynb`

Module Python principale:

- `ingestion.py`
- `preprocessing.py`
- `features.py`
- `baselines.py`
- `lightgbm_model.py`
- `nbeats_model.py`
- `evaluation.py`
- `interval_evaluation.py`
- `error_analysis.py`
- `config.py`

Testele se află în `tests/` și folosesc importuri prin `energy_load_forecast.<module>`, fără un `conftest.py` folosit doar pentru modificarea path-ului.

## 11. Reproductibilitate

Proiectul folosește `uv` pentru gestionarea mediului și a dependențelor. Repository-ul include `uv.lock`, astfel încât versiunile dependențelor sunt blocate.

Fluxul final trebuie rulat în această ordine:

1. obține datele ENTSO-E sau folosește CSV-ul orar din cache;
2. rulează notebook-urile de preprocessing și EDA, după necesitate;
3. rulează notebook-urile de modele sau notebook-ul de rezultate finale;
4. așteaptă finalizarea antrenării N-BEATS;
5. verifică `docs/results.md` și figurile salvate în `docs/`.

Orizontul final de test trebuie să conțină exact 24 de observații orare.

## 12. Checklist cerințe

| Cerință | Status |
|---|---|
| Problema de prognoză definită clar | Complet |
| Date istorice de consum colectate | Complet |
| Curățare și aliniere date | Complet |
| EDA pentru sezonalitatea zilnică și săptămânală | Complet |
| Discuție privind sezonalitatea anuală | Completă, cu limitarea setului de date menționată |
| Feature engineering | Complet |
| Împărțire cronologică train/validation/test | Complet |
| Baseline-uri simple | Complet |
| Mai multe modele candidate | Complet |
| MAE/RMSE/MAPE | Complet |
| Evaluarea intervalelor de predicție | Complet |
| Importanța caracteristicilor și analiza erorilor | Complet |
| Livrabil vizual final | Complet |
