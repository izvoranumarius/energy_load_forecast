[English](README.md)

# Prognoza consumului de energie electrică

Proiect de prognoză a consumului de energie electrică pentru România, realizat pe baza datelor istorice furnizate de ENTSO-E Transparency Platform.

## Despre proiect

Scopul proiectului este prognozarea consumului de energie electrică pentru România (`RO`), la nivel orar, pentru următoarele 24 de ore.

Fluxul final de lucru include:

- baseline Naive
- baseline Seasonal Naive
- modelul LightGBM
- modelul N-BEATS
- intervale de predicție P10-P90 folosind modele LightGBM pe cuantile
- MAE, RMSE și MAPE pentru prognozele punctuale
- pinball loss, grad de acoperire și lățimea medie a intervalului pentru prognozele de interval
- importanța variabilelor și analiza reziduurilor și a erorilor

Evaluarea finală folosește o împărțire cronologică în seturi de antrenare, validare și test. Setul de test conține 24 de observații orare.

## Date

Datele istorice privind consumul de energie electrică sunt preluate din ENTSO-E Transparency Platform.

Proiectul poate utiliza:

- fișiere CSV exportate din ENTSO-E Transparency Platform
- API-ul ENTSO-E prin biblioteca `entsoe-py`

Marcajele temporale sunt aduse la UTC, iar datele sunt agregate la o frecvență de o oră.

Experimentul final folosește date din perioada iulie 2025 - decembrie 2025. Deoarece perioada acoperă doar o parte dintr-un an, sezonalitatea anuală este tratată ca o limitare a setului de date și nu este estimată ca un tipar anual complet.

## Structura proiectului

```text
energy_load_forecast/
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
├── README.ro.md
├── uv.lock
├── configs/
│   └── config.yaml
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── all_models_comparison_plot.png
│   ├── final_comparison_plot.png
│   ├── final_report.md
│   ├── final_report.ro.md
│   └── results.md
├── notebooks/
│   ├── 00_explore_raw_csv.ipynb
│   ├── 01_api_ingestion.ipynb
│   ├── 02_preprocessing_exploration.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_baselines.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_lightgbm.ipynb
│   ├── 07_nbeats.ipynb
│   ├── 08_error_analysis.ipynb
│   └── 09_final_results.ipynb
├── src/
│   └── energy_load_forecast/
│       ├── __init__.py
│       ├── config.py
│       ├── error_analysis.py
│       ├── evaluation.py
│       ├── features.py
│       ├── ingestion.py
│       ├── interval_evaluation.py
│       ├── preprocessing.py
│       └── models/
└── tests/
    ├── test_baselines.py
    ├── test_error_analysis.py
    ├── test_evaluation.py
    ├── test_features.py
    ├── test_ingestion.py
    ├── test_interval_evaluation.py
    ├── test_lightgbm_model.py
    ├── test_nbeats_model.py
    └── test_preprocessing.py
```

Notebook-urile păstrează etapele de analiză și experimentare ale proiectului. Modulele Python din `src/` conțin implementarea reutilizabilă rezultată din această activitate.

## Mediu și instalare

Proiectul folosește [uv](https://docs.astral.sh/uv/) pentru gestionarea mediului și a dependențelor.

Pentru instalarea proiectului și a tuturor dependențelor din lockfile:

```bash
uv sync
```

Este necesară Python 3.14 sau o versiune mai nouă.

Pachetele necesare pentru dezvoltare, inclusiv `pytest`, `ruff` și `ipykernel`, sunt instalate prin `uv sync`.

## Cheia API ENTSO-E

Pentru utilizarea API-ului ENTSO-E, creează un fișier `.env` pornind de la `.env.example` și adaugă cheia API:

```text
ENTSOE_API_KEY=your_key_here
```

Fișierul `.env` și cheia API nu trebuie încărcate în repository.

## Rularea testelor

Pentru rularea întregii suite de teste:

```bash
uv run pytest
```

Testele acoperă ingestia datelor, preprocesarea, crearea variabilelor, baseline-urile, modelele, evaluarea prognozelor, evaluarea intervalelor și analiza erorilor.

## Rularea notebook-urilor

Pornește Jupyter Lab din mediul proiectului:

```bash
uv run jupyter lab
```

Notebook-urile urmează etapele proiectului:

1. `00_explore_raw_csv.ipynb` analizează datele brute în format CSV.
2. `01_api_ingestion.ipynb` tratează preluarea datelor prin API-ul ENTSO-E.
3. `02_preprocessing_exploration.ipynb` verifică prelucrarea și curățarea datelor.
4. `03_eda.ipynb` conține analiza exploratorie a datelor și analiza sezonalității.
5. `04_baselines.ipynb` evaluează modelele de referință.
6. `05_feature_engineering.ipynb` construiește variabilele calendaristice, lag-urile și mediile mobile.
7. `06_lightgbm.ipynb` dezvoltă și evaluează modelul LightGBM.
8. `07_nbeats.ipynb` dezvoltă și evaluează modelul N-BEATS.
9. `08_error_analysis.ipynb` analizează erorile de prognoză și comportamentul modelelor.
10. `09_final_results.ipynb` execută experimentul final și generează rezultatele și graficele principale.

## Experimentul final

După configurarea mediului și a datelor, rulează:

```text
notebooks/09_final_results.ipynb
```

Notebook-ul final:

1. încarcă sau preia seria orară de consum pentru România;
2. construiește seturile de antrenare, validare și test în ordine cronologică;
3. evaluează baseline-urile Naive și Seasonal Naive;
4. antrenează și evaluează LightGBM;
5. evaluează intervalul de predicție P10-P90 pentru LightGBM;
6. antrenează și evaluează N-BEATS;
7. salvează metricile finale în `docs/results.md`;
8. salvează graficele de comparație și diagnostic în `docs/`.

Pentru LightGBM, prognoza pe setul de test este realizată recursiv. Astfel, valorile reale viitoare ale consumului nu sunt folosite pentru construirea lag-urilor sau a mediilor mobile în timpul prognozei.

Antrenarea N-BEATS poate dura mai mult decât antrenarea celorlalte modele. Rezultatele finale pentru acest model trebuie preluate dintr-o rulare completă a notebook-ului `09_final_results.ipynb`.

## Rezultate finale

Toate modelele sunt evaluate pe aceeași fereastră de test de 24 de ore.

Metricile finale sunt salvate în:

```text
docs/results.md
```

Graficele principale sunt salvate în:

```text
docs/all_models_comparison_plot.png
docs/final_comparison_plot.png
```

Raportul final este disponibil în ambele versiuni:

```text
docs/final_report.md
docs/final_report.ro.md
```

## Reproductibilitate

Repository-ul conține `uv.lock`, astfel încât versiunile dependențelor sunt păstrate pentru reproducerea mediului.

Pornind de la o clonare nouă a repository-ului:

```bash
uv sync
uv run pytest
uv run jupyter lab
```

După pornirea Jupyter Lab, deschide `notebooks/09_final_results.ipynb` și rulează celulele în ordine.

Fereastra finală de test trebuie să conțină exact 24 de observații orare.
